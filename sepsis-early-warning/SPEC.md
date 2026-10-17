# Spec — Sepsis Early Warning (v1.0)

Spesifikasi ini ditulis sebelum implementasi, seperti PRD, lalu diperbarui dengan keputusan akhir. Hasil evaluasi lengkap ada di [README](README.md).

## 1. Masalah
UGD rumah sakit klien butuh peringatan dini untuk pasien yang berisiko sepsis dalam 24 jam, supaya dokter bisa memprioritaskan dan memulai perawatan lebih cepat.

## 2. Output yang diminta soal
| Bagian | Hasil |
|---|---|
| 1. Modeling | Preprocessing, model A (XGBoost), model B (ANN PyTorch), soft-voting berbobot, k-fold CV (AUC-ROC, precision, recall), simpan artefak |
| 2. API | FastAPI, endpoint `POST /predict`, input JSON satu pasien, output `{"sepsis_risk_prediction": 1, "risk_probability": 0.78}` |
| 3. Docker | `Dockerfile` + `requirements.txt` yang bisa jalan mandiri |
| 4. README | Pendekatan, performa, cara build/run Docker, contoh `curl`, keputusan desain dan asumsi |

## 3. Temuan dari data
Hasil profiling `data/sepsis_emr_data.csv`:

| Temuan | Detail | Dampak |
|---|---|---|
| Ukuran | 1.753 baris, 7 fitur, tanpa duplikat `patient_id` | — |
| Kelas | 1.017 non-sepsis (58%) vs 736 sepsis (42%) | Cukup seimbang, tidak perlu SMOTE |
| Nilai kosong | Hanya `lactate_level`: 141 baris (8%). **Semuanya pasien non-sepsis** (14% dari kelas 0, 0% dari kelas 1) | Kosongnya data sendiri jadi sinyal (lihat 4.3) |
| **Baris kembar** | Hanya **100 kombinasi fitur unik**, masing-masing muncul 17–18 kali (1.753 baris). Tidak ada label yang bertentangan | **Risiko data leakage besar** (lihat 4.1) |
| Outlier | `heart_rate` maksimum 220, pada satu profil non-sepsis | Dibiarkan dan dicatat; profil ini salah diprediksi sepsis dalam CV |
| Sinyal | Rata-rata semua fitur jauh berbeda antara kelas 0 dan 1 | Masalahnya mudah dipisahkan, skor tinggi wajar |

## 4. Keputusan desain
1. **Cross-validation dikelompokkan (`StratifiedGroupKFold`, k=5)** per kombinasi fitur. Dengan k-fold biasa, baris kembar yang sama jatuh di data latih *dan* validasi, sehingga AUC XGBoost menjadi 1,000 (model menghafal). Dengan CV terkelompok: 0,962.
2. **Preprocessing di-fit hanya pada data latih tiap fold:** imputasi median untuk `lactate_level`, lalu `StandardScaler` (penting untuk ANN, tidak merugikan pohon).
3. **Penanda `lactate_missing` tidak dipakai.** Di data ini kosong = tidak sepsis. Di dunia nyata, laktat memang sering tidak diperiksa pada pasien berisiko rendah, tetapi kebiasaan itu bisa berbeda di rumah sakit lain, sehingga sinyalnya berisiko menyesatkan. AUC ensemble dengan dan tanpa penanda sama (0,9975).
4. **Bobot ensemble 50/50.** Bobot XGBoost 0,0–1,0 diuji pada probabilitas out-of-fold; AUC datar untuk 0,2–0,9, jadi dipilih yang paling dekat 50/50.
5. **Threshold 0,5.** Recall lebih penting daripada precision untuk peringatan dini, tetapi threshold 0,3 tidak mengubah hasil pada data ini karena kesalahan berada pada probabilitas jauh di bawah 0,3.
6. **ANN dengan PyTorch (CPU)**, bukan TensorFlow/Keras, supaya image Docker lebih ringan.
7. **Validasi input API**: Pydantic menolak nilai tak masuk akal (misal suhu 80 °C) dengan HTTP 422; `lactate_level` boleh kosong.

## 5. Kriteria selesai
- AUC-ROC, precision, recall rata-rata dan simpangan dari CV terkelompok dilaporkan di README.
- API punya test `pytest` (6 test) dan smoke test Docker (`/health`, `/predict`) yang berjalan otomatis di GitHub Actions.

## 6. Di luar cakupan
Autentikasi, database, monitoring produksi, dan validasi klinis nyata. Dicatat sebagai keterbatasan di README, karena data ini tampak sintetis.
