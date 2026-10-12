# Structural Damage Classification

Proyek Tugas 1 Supervised Machine Learning untuk memprediksi `damage_grade` bangunan
pasca gempa ke dalam lima kelas. Model final memakai LightGBM, bukan Neural Network.

- Nama: **Rahardian Putra**
- Kompetisi: [Structural Damage Classification](https://www.kaggle.com/competitions/structural-damage-classification)
- Metrik: **Macro F1-Score**
- Tenggat kompetisi: **14 Oktober 2026, 20.00 WIB**
- Public score V1: **0,55165**
- Public score V2: **0,56747**
- Private score V2: **belum tersedia sebelum kompetisi berakhir**

## Pemenuhan ketentuan

| Ketentuan | Implementasi |
|---|---|
| Tanpa Neural Network | Dummy, Logistic Regression, Random Forest, dan LightGBM |
| `Pipeline` | Feature engineering, preprocessing, dan estimator berada dalam Pipeline |
| `ColumnTransformer` | Cabang numerik dan kategorikal diproses secara terpisah |
| Pencegahan leakage | Encoder dan imputer hanya di-fit pada bagian training setiap split |
| Docstring | Seluruh function dan class di `scripts/` memiliki docstring |
| Source modular | Implementasi reusable berada di `scripts/` |
| Notebook runtut | EDA, baseline, pemodelan, evaluasi, refit, dan submission |
| Interpretasi | Perbandingan model, confusion matrix, error analysis, dan keterbatasan tersedia di bawah |

## Struktur repository

```text
structural-damage-classification/
├── README.md
├── FINAL_CHECKLIST.md
├── requirements.txt
├── .gitignore
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_submission.ipynb
│   ├── 04_final_gpu_v2.ipynb
│   └── structural_damage_classification.ipynb
├── scripts/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── models/
├── submissions/
└── reports/
    └── figures/
```

Dataset tidak disimpan di GitHub. Aturan `.gitignore` mengecualikan `data/*.csv`,
arsip data, kredensial Kaggle, cache Python, model Joblib, dan file sistem operasi.

## Menjalankan notebook final di Kaggle

1. Bergabung dengan kompetisi dan menerima rules.
2. Buka **Code -> New Notebook**.
3. Tambahkan data kompetisi melalui **+ Add Input**.
4. Buka **Settings -> Accelerator -> GPU T4 x2**.
5. Impor `notebooks/04_final_gpu_v2.ipynb`.
6. Pastikan `FAST_MODE = False`, kemudian pilih **Restart Session -> Run All**.
7. Unduh `/kaggle/working/submission_improved_v2.csv`.
8. Simpan versi Kaggle dengan **Save & Run All** agar output eksekusi dapat diaudit.

Notebook memeriksa keberadaan dua GPU. LightGBM 4.6 pada runtime Kaggle memakai
backend OpenCL: model yang berbeda dilatih paralel pada GPU 0 dan GPU 1. Kategori
dengan kardinalitas tinggi diproses sebagai ordinal numerik supaya tidak melampaui
batas bin GPU; kategori kecil tetap diberikan sebagai kategori native LightGBM.

## Menjalankan secara lokal

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.train --data-dir data --output-dir .
```

Letakkan `train.csv`, `test.csv`, dan `sample_submission.csv` di `data/` terlebih
dahulu. Mode modular lokal mereproduksi workflow dasar berbasis CPU. Notebook
`04_final_gpu_v2.ipynb` adalah implementasi final yang dipakai untuk hasil V2.

## Data dan validasi

- Train: **533.465 baris, 17 kolom**.
- Test: **228.629 baris, 16 kolom**.
- Target: `damage_grade`, yaitu `Grade 1` sampai `Grade 5`.
- Tidak ada missing value dan `building_id` train/test tidak beririsan.
- Split utama: development 80% dan holdout 20%, stratified, `random_state=42`.
- Development dibagi lagi menjadi fit dan validation untuk memilih profil fitur,
  bobot blend, dan multiplier kelas.
- Holdout tidak digunakan ketika memilih konfigurasi dan hanya dievaluasi setelah
  konfigurasi final ditetapkan.

Test kompetisi hanya digunakan untuk menghasilkan prediksi. Label test tidak pernah
digunakan dalam preprocessing, feature engineering, tuning, atau pemilihan model.

## Feature engineering dan preprocessing

Fitur dasar mencakup karakteristik bangunan, material, geometri, dan lokasi. Pipeline
menambahkan fitur berikut tanpa memakai target:

- `height_per_floor`;
- `estimated_floor_area`;
- `age_is_999`;
- `log_plinth_area`;
- potongan dua digit `building_id`;
- prefix hierarkis 4, 6, dan 8 digit serta suffix 2 digit.

Cabang numerik menggunakan imputasi median. Cabang kategorikal menggunakan imputasi
modus dan `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
Untuk GPU, kategori dengan paling banyak 128 level tetap kategori native; `ward_id`
dan prefix yang lebih besar dianggap numerik sehingga dibatasi `max_bin=63`.

## Eksperimen

### Baseline

| Model | Baris latih | Macro F1 |
|---|---:|---:|
| Dummy prior | 80.000 | 0,1063 |
| Logistic Regression | 80.000 | 0,5040 |
| Random Forest V1 | 120.000 | 0,5027 |
| LightGBM V1 | 426.772 | 0,5483 |

Baseline memakai data lebih sedikit untuk menjaga waktu eksperimen. Angkanya tidak
boleh dibandingkan tanpa memperhatikan jumlah data latih.

### LightGBM V2

V2 membandingkan dua profil:

| Profil | Validation Macro F1 |
|---|---:|
| Segments | 0,5386 |
| Hierarchical | **0,5574** |

Pencarian bobot memilih profil hierarchical sepenuhnya (`weight=1,0`). Multiplier
kelas dipilih hanya pada validation internal. Pada run final yang dilaporkan,
validation Macro F1 mencapai **0,561260**. Dua evaluasi GPU pada holdout menghasilkan
**0,5609** dan **0,5621**; variasi kecil ini berasal dari sifat komputasi GPU.
Nilai terbaru berdasarkan confusion matrix lengkap adalah **0,562109**.

Peningkatan terhadap V1:

```text
0,562109 - 0,548300 = +0,013809 Macro F1
```

Model final dilatih ulang pada seluruh train menggunakan dua seed, 42 dan 179.
Submission V2 berisi 228.629 baris dengan kolom `building_id,damage_grade` dan label
numerik 1-5 sesuai evaluator kompetisi.

## Evaluasi per kelas V2

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Grade 1 | 0,659 | 0,692 | 0,675 | 11.034 |
| Grade 2 | 0,384 | 0,482 | 0,428 | 12.216 |
| Grade 3 | 0,438 | 0,447 | 0,443 | 19.098 |
| Grade 4 | 0,519 | 0,555 | 0,536 | 25.738 |
| Grade 5 | 0,794 | 0,674 | 0,729 | 38.607 |
| **Macro average** | **0,559** | **0,570** | **0,562** | **106.693** |

Grade 1 dan Grade 5 paling mudah dipisahkan. Grade 2 dan Grade 3 tetap menjadi kelas
tersulit karena karakteristik kerusakannya bertumpang tindih dengan kelas di sebelahnya.
Confusion matrix menunjukkan sebagian besar kesalahan terjadi antargrade yang berdekatan,
terutama Grade 2 dengan Grade 1/3 dan Grade 4 dengan Grade 3/5.

Lokasi adalah sinyal utama. Prefix hierarkis `building_id`, `ward_id`, dan `vdcmun_id`
membantu model menangkap konsentrasi kerusakan secara spasial. Hasil ini tidak berarti
model akan memiliki performa sama pada wilayah baru yang tidak terdapat di train.

## Submission dan skor Kaggle

Submission V1 diterima Kaggle dengan public Macro F1 **0,55165**. Submission V2
diterima dengan public Macro F1 **0,56747**, meningkat **0,01582**. Keduanya adalah
skor leaderboard nyata dan berbeda dari holdout lokal. File V2 lolos pemeriksaan
bentuk, tipe label, urutan ID, nilai kosong, dan model save/load round-trip. Private
score belum tersedia karena kompetisi masih berlangsung.

Private leaderboard memakai bagian test yang lebih besar, sehingga keputusan model
utama didasarkan pada validation dan holdout, bukan hanya perubahan public score.

## Keterbatasan

- GPU OpenCL tidak mendukung categorical bins yang sangat besar; ID besar harus
  direpresentasikan secara ordinal dan dibatasi menjadi 63 bin.
- Split acak mengukur bangunan baru pada wilayah yang sebagian besar sudah dikenal.
- Data tidak menyertakan seluruh detail material yang disebutkan metadata asli.
- Grade 2 dan Grade 3 masih memiliki precision dan F1 yang relatif rendah.
- LightGBM GPU tidak sepenuhnya deterministik, sehingga run ulang dapat berbeda kecil.

## Artefak hasil

- `reports/results_v2.json`: ringkasan run V2 dan status leaderboard.
- `reports/model_comparison_v2.csv`: perbandingan baseline dan model final.
- `reports/classification_report_v2.csv`: metrik per kelas.
- `reports/confusion_matrix_v2.csv`: nilai confusion matrix.
- `reports/figures/confusion_matrix_v2.png`: visualisasi kesalahan V2.
- `notebooks/04_final_gpu_v2.ipynb`: notebook final untuk Kaggle T4 x2.
