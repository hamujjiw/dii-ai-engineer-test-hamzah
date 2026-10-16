# Sepsis Early Warning API

Layanan prediksi risiko sepsis untuk pasien UGD. Menerima tanda vital dan hasil lab awal satu pasien (JSON), lalu mengembalikan prediksi dan probabilitas dari **ensemble soft-voting XGBoost + ANN (PyTorch)**.

## Pendekatan singkat
1. **Preprocessing** (di-fit hanya pada data latih): nilai kosong `lactate_level` diisi median, lalu `StandardScaler`.
2. **Model A:** XGBoost (200 pohon, kedalaman 3). **Model B:** ANN/MLP PyTorch (32 → 16 → 1 neuron).
3. **Ensemble:** probabilitas kedua model dirata-rata berbobot (`w·XGB + (1−w)·ANN`).
4. **Evaluasi:** 5-fold `StratifiedGroupKFold` (lihat bagian *Keputusan desain* untuk alasannya).
5. **Artefak:** `artifacts/` berisi preprocessor, model XGBoost, model ANN, dan ensemble gabungan.

## Performa (5-fold CV terkelompok, threshold 0,5)
| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost saja | 0,962 ± 0,085 | 0,958 ± 0,058 | 0,924 ± 0,169 | 0,931 ± 0,096 |
| ANN saja | 0,989 ± 0,026 | 0,958 ± 0,058 | 0,949 ± 0,115 | 0,948 ± 0,060 |
| **Ensemble 50/50** | **1,000 ± 0,000** (per fold) / 0,9975 (gabungan semua fold) | 0,958 ± 0,058 | 0,924 ± 0,169 | 0,931 ± 0,096 |

Angka ± adalah simpangan antar fold. Satu fold hanya berisi ±20 profil pasien, jadi simpangannya besar: hasil ini **indikatif, bukan jaminan performa klinis**. Detail lengkap ada di `artifacts/metrics.json`.

## Cara menjalankan dengan Docker
Butuh [Docker Desktop](https://www.docker.com/products/docker-desktop/). Jalankan dari folder `sepsis-early-warning`:

```bash
docker build -t sepsis-api .
docker run --rm -p 8000:8000 sepsis-api
```

Cek bahwa layanan hidup, lalu buka dokumentasi interaktif di http://localhost:8000/docs.

```bash
curl http://localhost:8000/health
```

## Contoh permintaan
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"heart_rate":122,"respiratory_rate":29,"temperature":39.2,"wbc_count":18.1,"lactate_level":4.7,"age":83,"num_comorbidities":4}'
```
Respons:
```json
{"sepsis_risk_prediction": 1, "risk_probability": 0.9992}
```
`lactate_level` boleh `null` atau dihilangkan (laktat tidak selalu diperiksa). Nilai yang mustahil secara fisiologis (misalnya `temperature: 80`) ditolak dengan HTTP 422.

Contoh dengan Python:
```python
import requests
r = requests.post("http://localhost:8000/predict", json={
    "heart_rate": 83, "respiratory_rate": 15, "temperature": 36.8,
    "wbc_count": 6.8, "lactate_level": 1.0, "age": 36, "num_comorbidities": 1})
print(r.json())
```

## Menjalankan tanpa Docker / melatih ulang
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements-dev.txt
python -m sepsis_ew.train                           # latih ulang + simpan artefak
pytest                                              # jalankan test API
uvicorn api.main:app --reload                       # jalankan API
```

## Struktur repo
```
sepsis_ew/models.py   preprocessing, ANN PyTorch, kelas ensemble
sepsis_ew/train.py    training, CV, pemilihan bobot, simpan artefak
api/main.py           FastAPI (/predict, /health)
artifacts/            model terlatih + metrics.json
tests/test_api.py     test API
Dockerfile, requirements.txt
```

## Keputusan desain dan asumsi
- **CV terkelompok, bukan k-fold biasa.** Dari 1.753 baris ternyata hanya ada **100 profil pasien unik**, masing-masing muncul 17–18 kali. Dengan k-fold biasa, profil yang sama masuk ke data latih *dan* validasi: AUC XGBoost menjadi 1,000 (model hanya menghafal). Dengan `StratifiedGroupKFold` (satu profil selalu berada di satu sisi), AUC turun ke 0,962, angka yang lebih jujur.
- **Bobot ensemble 50/50.** AUC gabungan lintas fold nyaris datar untuk bobot XGBoost 0,2–0,9 (0,9975), sementara tiap model sendirian lebih buruk (ANN 0,987; XGBoost 0,947). Karena tidak ada bobot yang jelas lebih baik, dipilih yang paling dekat 50/50 (aturan: bila selisih AUC < 0,001, pilih yang terdekat 0,5). Ini lebih sederhana dan tidak terlalu "menyetel" ke data sekecil ini.
- **Kolom `lactate_level` kosong hanya terjadi pada pasien non-sepsis** (14% kelas 0, 0% kelas 1). Penanda "laktat kosong" sengaja **tidak dipakai** sebagai fitur: di rumah sakit lain kebiasaan pemeriksaan laktat bisa berbeda, sehingga sinyal itu berisiko menyesatkan. Perbandingan AUC kedua opsi hampir sama (ensemble 0,9975 pada keduanya).
- **Threshold 0,5.** Untuk sistem peringatan dini, recall (jangan sampai pasien sepsis terlewat) lebih penting daripada precision. Pada data ini menurunkan threshold ke 0,3 tidak mengubah hasil, karena kesalahan terjadi pada kasus batas yang probabilitasnya jauh di bawah 0,3 (lihat berikut). Threshold bisa diubah di `SoftVotingEnsemble.threshold`.
- **Kesalahan model terkonsentrasi pada kasus batas.** Dalam CV terkelompok ada 3 profil sepsis (HR 105–106, napas 22, suhu 38,1–38,2 °C) yang diprediksi rendah karena sangat mirip dengan satu profil non-sepsis (HR 104, napas 22, suhu 38,0 °C), serta satu profil non-sepsis dengan `heart_rate = 220` (kemungkinan salah input) yang diprediksi sepsis.
- **Data tampak sintetis** (profil berulang, pemisahan kelas sangat tegas), jadi skor tinggi tidak boleh dibaca sebagai performa di rumah sakit nyata. Layanan ini bukan alat diagnosis; keputusan klinis tetap pada dokter.
- **Keterbatasan:** tanpa autentikasi, logging, atau monitoring drift; belum divalidasi klinis; kalibrasi probabilitas belum diperiksa.
