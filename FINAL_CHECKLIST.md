# Final Checklist

## Repository

- [x] Struktur `data/`, `notebooks/`, `scripts/`, `models/`, `submissions/`, dan `reports/` tersedia.
- [x] Dataset dan kredensial Kaggle dikecualikan oleh `.gitignore`.
- [x] README menjelaskan cara bergabung dan mengunduh dataset melalui web atau Kaggle API.
- [x] Folder `models/` dilacak dengan `.gitkeep`; artefak model dihasilkan saat training.
- [x] Seluruh function/class di `scripts/` memiliki docstring.
- [x] Notebook memakai `Pipeline` dan `ColumnTransformer`.
- [x] Model final bukan Neural Network.
- [x] README memuat cara menjalankan, validasi, hasil, interpretasi, dan keterbatasan.

## Notebook final

- [x] Data kompetisi ditemukan dari `/kaggle/input`.
- [x] GPU T4 x2 dan LightGBM OpenCL terdeteksi.
- [x] Baseline dijalankan.
- [x] Validation dan holdout dipisahkan.
- [x] Holdout Macro F1 terbaru: 0,562109.
- [x] Model final dilatih ulang pada seluruh train dengan seed 42 dan 179.
- [x] Model save/load round-trip terverifikasi.
- [x] Submission 228.629 baris berhasil dibuat pada Kaggle.

## Sebelum dikumpulkan

- [x] Salin hasil Kaggle sebagai `submissions/submission_v2_public_0.56747.csv`.
- [x] Submission V2 diterima Kaggle dengan public Macro F1 0,56747.
- [x] Tambahkan public score V2 ke `reports/results_v2.json` dan README.
- [ ] Unduh notebook Kaggle yang sudah selesai dieksekusi dan ganti `04_model_final_lightgbm_gpu_t4x2.ipynb` jika dosen meminta output tersimpan.
- [x] Identitas Rahardian Putra dan NRP 5052231018 tercantum pada README serta notebook.
- [x] Push seluruh source, notebook, reports, dan submission ke GitHub.
- [x] Repository GitHub bersifat publik dan dapat dibuka oleh dosen.
