# Spec — Sepsis Early Warning (draft v0.1)

Dokumen ini dibuat sebelum ngoding, seperti PRD. Nanti bagian-bagiannya jadi kerangka README dan bahan presentasi.

## 1. Masalah
UGD rumah sakit klien butuh peringatan dini untuk pasien yang berisiko sepsis dalam 24 jam, supaya dokter bisa memprioritaskan dan memulai perawatan lebih cepat.

## 2. Output yang diminta soal
| Bagian | Hasil |
|---|---|
| 1. Modeling | Preprocessing, model A (XGBoost/LightGBM), model B (ANN), soft-voting berbobot, k-fold CV (AUC-ROC, precision, recall), simpan artefak |
| 2. API | FastAPI, satu endpoint `POST /predict`, input JSON satu pasien, output `{"sepsis_risk_prediction": 1, "risk_probability": 0.78}` |
| 3. Docker | `Dockerfile` + `requirements.txt` yang bisa jalan mandiri |
| 4. README | Pendekatan, performa, cara build/run Docker, contoh `curl`, keputusan desain dan asumsi |

## 3. Temuan awal dari data
Hasil profiling `data/sepsis_emr_data.csv`:

| Temuan | Detail | Dampak |
|---|---|---|
| Ukuran | 1.753 baris, 7 fitur, tanpa duplikat `patient_id` | — |
| Kelas | 1.017 non-sepsis (58%) vs 736 sepsis (42%) | Cukup seimbang, tidak perlu SMOTE |
| Nilai kosong | Hanya `lactate_level`: 141 baris (8%). **Semuanya pasien non-sepsis** (14% dari kelas 0, 0% dari kelas 1) | Kosongnya data sendiri jadi sinyal. Perlu diputuskan cara menanganinya |
| **Baris kembar** | Hanya **100 kombinasi fitur unik**, masing-masing muncul 17–18 kali (1.753 baris). Tidak ada label yang bertentangan | **Risiko data leakage besar** (lihat 4) |
| Outlier | `heart_rate` maksimum 220 | Cek apakah satu profil pasien atau salah input |
| Sinyal | Rata-rata semua fitur jauh berbeda antara kelas 0 dan 1 | Masalahnya mudah dipisahkan, skor tinggi wajar |

## 4. Keputusan desain (usulan, untuk dikonfirmasi)
1. **Cross-validation dikelompokkan (`StratifiedGroupKFold`)** per kombinasi fitur. Kalau pakai k-fold biasa, baris kembar yang sama jatuh di data latih *dan* validasi, sehingga skor kelihatan sempurna padahal model cuma menghafal. Ini temuan terpenting untuk dibahas di interview.
2. **Preprocessing di dalam Pipeline**, di-fit hanya pada data latih tiap fold: imputasi median untuk `lactate_level` + kolom penanda `lactate_missing`, lalu `StandardScaler` (penting untuk ANN, tidak berpengaruh ke pohon).
3. **`lactate_missing` dipertahankan atau dibuang?** Di data ini kosong = tidak sepsis. Di dunia nyata, laktat sering tidak diperiksa pada pasien yang dinilai berisiko rendah, jadi sinyalnya masuk akal secara klinis, tapi bisa menyesatkan kalau alat ukur di rumah sakit lain berbeda. Akan dibandingkan kedua opsi dan dijelaskan di README.
4. **Bobot ensemble** dipilih dari hasil CV terkelompok (bukan asal 50/50), lalu dijelaskan.
5. **Threshold**: sistem peringatan dini lebih mementingkan **recall** (jangan sampai pasien sepsis lolos) daripada precision. Akan dilaporkan hasil di threshold 0.5 dan threshold alternatif.
6. **ANN dengan PyTorch atau Keras** (diputuskan saat setup API supaya image Docker tidak terlalu besar).
7. **Validasi input API**: Pydantic menolak nilai tak masuk akal (misal suhu 80 °C) dengan pesan jelas.

## 5. Metrik sukses
- AUC-ROC, precision, recall rata-rata dan simpangan dari CV terkelompok.
- API membalas < 1 detik di satu container, dan punya test `pytest`.

## 6. Di luar cakupan
Autentikasi, database, monitoring produksi, dan validasi klinis nyata. Dicatat sebagai keterbatasan di README, karena data ini tampak sintetis.

## 7. Pertanyaan terbuka
- [ ] Bagian ke-4 di PDF ("empat bagian") tidak ada, tanyakan ke HRD.
- [ ] `lactate_missing` dipakai atau tidak (lihat 4.3).
- [ ] Keras atau PyTorch untuk ANN.
