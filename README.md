# Online Retail Recommendation System

Sistem rekomendasi produk untuk dataset UCI Online Retail (Des 2010 - Des 2011).

- **Metode utama:** Apriori (association rules) untuk rekomendasi berbasis keranjang.
- **Metode pendukung:** Item-Based Collaborative Filtering (implicit feedback, cosine similarity) untuk rekomendasi personal dan pelengkap Apriori.

Disusun oleh:
Ricky Hardian (NRP: 6032252067)
Mulyana (NRP: 6032252055)

## Struktur
- `notebooks/` notebook analisis Tahap 0-13
- `app/app.py` dashboard Streamlit
- `app/artifacts/` hasil pra-komputasi yang dibaca dashboard

## Menjalankan lokal
```
pip install -r requirements.txt
streamlit run app/app.py
```

## Dashboard
[URL dashboard]
