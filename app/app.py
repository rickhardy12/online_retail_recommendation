import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Online Retail Recommendation System", layout="wide")

ART = Path(__file__).resolve().parent / "artifacts"
BLUE, TEAL, GREY = "#1F6FB2", "#0F8B8D", "#94A3B8"
TEMPLATE = "plotly_white"
PAGES = [
    "Ringkasan",
    "Market Basket Explorer",
    "Rekomendasi Keranjang",
    "Produk Serupa",
    "Rekomendasi Customer",
    "Evaluasi Model",
]


@st.cache_data
def load_artifacts():
    rules = pd.read_parquet(ART / "rules.parquet")
    rules["ant_set"] = rules["antecedents"].str.split(" | ", regex=False).apply(frozenset)
    rules["con_set"] = rules["consequents"].str.split(" | ", regex=False).apply(frozenset)
    data = {
        "rules": rules,
        "nbr": pd.read_parquet(ART / "cf_item_neighbors.parquet"),
        "lookup": pd.read_parquet(ART / "product_lookup.parquet"),
        "hist": pd.read_parquet(ART / "customer_history.parquet"),
        "monthly": pd.read_csv(ART / "eda_monthly.csv"),
        "top": pd.read_csv(ART / "eda_top_products.csv"),
        "country": pd.read_csv(ART / "eda_country.csv"),
        "evaluation": pd.read_csv(ART / "evaluation_metrics.csv"),
        "meta": json.loads((ART / "metadata.json").read_text()),
    }
    return data


D = load_artifacts()
RULES, NBR, LOOKUP, HIST, META = D["rules"], D["nbr"], D["lookup"], D["hist"], D["meta"]
DESC = dict(zip(LOOKUP["StockCode"], LOOKUP["Description"]))


def name(code):
    return DESC.get(code, code)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def recommend_apriori(basket, top_n=10):
    basket = frozenset(basket)
    hit = RULES[RULES["ant_set"].apply(lambda a: a <= basket)]
    rows = []
    for r in hit.itertuples():
        for c in r.con_set - basket:
            rows.append({
                "StockCode": c,
                "score": r.confidence * r.lift,
                "confidence": r.confidence,
                "lift": r.lift,
                "rule_sumber": f"{r.antecedents_desc} → {r.consequents_desc}",
            })
    if not rows:
        return pd.DataFrame(columns=["StockCode", "score", "confidence", "lift", "rule_sumber"])
    out = pd.DataFrame(rows).sort_values("score", ascending=False).drop_duplicates("StockCode")
    return out.head(top_n)


def recommend_cf_basket(basket, exclude, top_n=10):
    sub = NBR[NBR["StockCode"].isin(basket)]
    sub = sub[~sub["neighbor_StockCode"].isin(exclude)]
    if sub.empty:
        return pd.DataFrame(columns=["StockCode", "score"])
    s = sub.groupby("neighbor_StockCode")["similarity"].sum().sort_values(ascending=False).head(top_n)
    return s.rename("score").rename_axis("StockCode").reset_index()


def recommend_integrated(basket, top_n=10):
    ap = recommend_apriori(basket, top_n).assign(sumber="Apriori")
    need = top_n - len(ap)
    parts = [ap]
    if need > 0:
        cf = recommend_cf_basket(basket, set(basket) | set(ap["StockCode"]), need).assign(sumber="Item-CF")
        parts.append(cf)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "Produk", out["StockCode"].map(name))
    return out


def recommend_cf_customer(customer_id, top_n=10):
    h = HIST[HIST["CustomerID"] == customer_id][["StockCode", "invoices"]]
    w = h.assign(w=np.log1p(h["invoices"]))[["StockCode", "w"]]
    m = NBR.merge(w, left_on="neighbor_StockCode", right_on="StockCode", how="inner", suffixes=("", "_n"))
    num = (m["similarity"] * m["w"]).groupby(m["StockCode"]).sum()
    den = NBR.groupby("StockCode")["similarity"].sum()
    score = (num / den.reindex(num.index)).drop(labels=h["StockCode"], errors="ignore")
    score = score.sort_values(ascending=False).head(top_n)
    out = score.rename("score").rename_axis("StockCode").reset_index()
    out.insert(0, "Produk", out["StockCode"].map(name))
    return out


def rules_network(r):
    g = nx.DiGraph()
    for row in r.itertuples():
        for a in row.ant_set:
            for c in row.con_set:
                g.add_edge(a, c, w=row.confidence)
    if g.number_of_nodes() == 0:
        return None
    pos = nx.spring_layout(g, seed=42, k=0.9)
    fig = go.Figure()
    for a, c, d in g.edges(data=True):
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[c][0]], y=[pos[a][1], pos[c][1]], mode="lines",
                                 line=dict(width=1 + 4 * d["w"], color=GREY), hoverinfo="skip", showlegend=False))
    nodes = list(g.nodes())
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=[name(n)[:25] for n in nodes], textposition="top center", textfont=dict(size=10),
        hovertext=[name(n) for n in nodes], hoverinfo="text",
        marker=dict(size=12, color=BLUE, line=dict(width=1, color="white")), showlegend=False))
    fig.update_layout(template=TEMPLATE, height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


st.sidebar.title("Online Retail Recommendation System")
page = st.sidebar.radio("Halaman", PAGES)
st.sidebar.markdown(
    f"**Cakupan model**  \nNegara: {META['scope_country']}  \n"
    f"Periode: {META['date_min']} s.d. {META['date_max']}"
)
st.sidebar.caption("Sumber data: UCI Online Retail Dataset")

if page == "Ringkasan":
    st.title("Ringkasan Data")
    c = st.columns(5)
    c[0].metric("Invoice", fmt_int(META["n_invoices"]))
    c[1].metric("Produk", fmt_int(META["n_products"]))
    c[2].metric("Customer", fmt_int(META["n_customers"]))
    c[3].metric("Revenue (GBP)", f"£{META['revenue'] / 1e6:,.2f} jt")
    c[4].metric("Rules Apriori", fmt_int(META["n_rules"]))
    m = D["monthly"]
    fig = px.bar(m, x="InvoiceMonth", y="revenue", template=TEMPLATE, color_discrete_sequence=[BLUE],
                 labels={"InvoiceMonth": "Bulan", "revenue": "Revenue (GBP)"}, title="Revenue bulanan")
    st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    t = D["top"].head(10).sort_values("invoices")
    left.plotly_chart(px.bar(t, x="invoices", y="Description", orientation="h", template=TEMPLATE,
                             color_discrete_sequence=[BLUE], title="10 produk paling sering dibeli",
                             labels={"invoices": "Jumlah invoice", "Description": ""}), width="stretch")
    cty = D["country"].head(10).sort_values("invoices")
    right.plotly_chart(px.bar(cty, x="invoices", y="Country", orientation="h", template=TEMPLATE,
                              color_discrete_sequence=[TEAL], title="10 negara teratas",
                              labels={"invoices": "Jumlah invoice", "Country": ""}), width="stretch")

elif page == "Market Basket Explorer":
    st.title("Market Basket Explorer")
    st.caption(f"Rules Apriori (min_support {META['min_support']}, min_confidence {META['min_confidence']}, "
               f"lift ≥ {META['min_lift']}) setelah penghapusan rules redundan.")
    c1, c2, c3, c4 = st.columns(4)
    s_min = c1.slider("Min support", float(RULES["support"].min()), float(RULES["support"].max()),
                      float(RULES["support"].min()), step=0.001, format="%.3f")
    conf_min = c2.slider("Min confidence", 0.0, 1.0, float(META["min_confidence"]), 0.05)
    lift_min = c3.slider("Min lift", 1.0, float(np.ceil(RULES["lift"].max())), 1.0, 0.5)
    q = c4.text_input("Cari produk", "").strip().upper()
    f = RULES[(RULES["support"] >= s_min) & (RULES["confidence"] >= conf_min) & (RULES["lift"] >= lift_min)]
    if q:
        f = f[f["antecedents_desc"].str.contains(q, regex=False) | f["consequents_desc"].str.contains(q, regex=False)]
    k = st.columns(3)
    k[0].metric("Rules lolos filter", fmt_int(len(f)))
    k[1].metric("Rata-rata confidence", f"{f['confidence'].mean():.2f}" if len(f) else "-")
    k[2].metric("Lift tertinggi", f"{f['lift'].max():.2f}" if len(f) else "-")
    if f.empty:
        st.info("Tidak ada rules yang lolos filter. Longgarkan filter di atas.")
    else:
        show = f[["antecedents_desc", "consequents_desc", "support", "confidence", "lift"]].sort_values("lift", ascending=False)
        st.dataframe(show.round(3), width="stretch", hide_index=True)
        st.download_button("Unduh CSV", show.to_csv(index=False).encode("utf-8"), "rules_terfilter.csv", "text/csv")
        left, right = st.columns(2)
        left.plotly_chart(px.scatter(f, x="support", y="confidence", color="lift", template=TEMPLATE,
                                     color_continuous_scale="Blues", hover_data=["antecedents_desc", "consequents_desc"],
                                     title="Support vs confidence"), width="stretch")
        net = rules_network(f.sort_values("lift", ascending=False).head(25))
        if net is not None:
            right.markdown("**Network 25 rules teratas (lift)**")
            right.plotly_chart(net, width="stretch")

elif page == "Rekomendasi Keranjang":
    st.title("Rekomendasi Keranjang")
    st.caption("Apriori menjadi sumber utama. Jika rules tidak cukup, slot sisa diisi tetangga Item-Based CF.")
    options = sorted(set(LOOKUP["StockCode"]) & (set(NBR["StockCode"]) | set().union(*RULES["ant_set"])), key=name)
    default = [c for c in RULES.sort_values("lift", ascending=False)["ant_set"].iloc[0]] if len(RULES) else []
    basket = st.multiselect("Produk di keranjang", options, default=default, format_func=name)
    top_n = st.slider("Jumlah rekomendasi", 3, 15, 8)
    if not basket:
        st.info("Pilih minimal satu produk.")
    else:
        rec = recommend_integrated(basket, top_n)
        if rec.empty:
            st.info("Belum ada rekomendasi untuk keranjang ini.")
        else:
            st.dataframe(rec[["Produk", "sumber", "score", "confidence", "lift", "rule_sumber"]].round(3),
                         width="stretch", hide_index=True)
            st.caption("Skor Apriori = confidence × lift. Skor Item-CF = jumlah cosine similarity terhadap produk di keranjang.")

elif page == "Produk Serupa":
    st.title("Produk Serupa (Item-Based CF)")
    pop = D["top"]["StockCode"].tolist()
    options = sorted(NBR["StockCode"].unique(), key=lambda c: (c not in pop, pop.index(c) if c in pop else 0, name(c)))
    code = st.selectbox("Produk acuan", options, format_func=name)
    sub = NBR[NBR["StockCode"] == code].sort_values("rank")
    if sub.empty:
        st.info("Produk ini belum memiliki tetangga.")
    else:
        sub = sub.assign(Produk=sub["neighbor_StockCode"].map(name))
        fig = px.bar(sub.sort_values("similarity"), x="similarity", y="Produk", orientation="h", template=TEMPLATE,
                     color_discrete_sequence=[TEAL], labels={"similarity": "Cosine similarity", "Produk": ""},
                     title="Tetangga terdekat")
        fig.update_layout(height=600)
        st.plotly_chart(fig, width="stretch")

elif page == "Rekomendasi Customer":
    st.title("Rekomendasi Customer")
    customers = sorted(HIST["CustomerID"].unique())
    cid = st.selectbox("CustomerID", customers)
    h = HIST[HIST["CustomerID"] == cid].sort_values("invoices", ascending=False)
    left, right = st.columns(2)
    left.markdown(f"**Riwayat pembelian** ({len(h)} produk unik)")
    left.dataframe(h.assign(Produk=h["StockCode"].map(name))[["Produk", "invoices"]].head(15),
                   width="stretch", hide_index=True)
    rec = recommend_cf_customer(cid, 10)
    right.markdown("**Rekomendasi Item-CF**")
    if rec.empty:
        right.info("Belum ada rekomendasi untuk customer ini.")
    else:
        right.dataframe(rec[["Produk", "score"]].round(3), width="stretch", hide_index=True)

else:
    st.title("Evaluasi Model")
    ev = D["evaluation"]
    st.caption("Evaluasi offline dengan split temporal. Model dilatih pada data sebelum tanggal split dan diuji setelahnya.")
    st.dataframe(ev.pivot_table(index=["skenario", "metrik"], columns="metode", values="nilai").round(4),
                 width="stretch")
    fig = px.bar(ev, x="metrik", y="nilai", color="metode", barmode="group", facet_col="skenario", template=TEMPLATE,
                 color_discrete_map={"Apriori": BLUE, "Item-CF": TEAL, "Popularity": GREY})
    fig.update_xaxes(matches=None)
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "- **HitRate@K**: persen kasus ketika produk target muncul di K rekomendasi teratas.\n"
        "- **Precision@K / Recall@K**: ketepatan dan kelengkapan K rekomendasi terhadap produk yang benar-benar dibeli.\n"
        "- **Coverage**: persen kasus yang mendapat minimal satu rekomendasi."
    )
