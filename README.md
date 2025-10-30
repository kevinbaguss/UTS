# Polynomial Regression UTS machine learning

Project ini melakukan **Polynomial Regression** untuk memprediksi harga properti berdasarkan beberapa fitur. Dataset dapat dibuat secara otomatis atau dimuat dari CSV yang sudah ada. Project ini juga membandingkan performa model Linear, Ridge, dan Lasso dengan berbagai degree polynomial dan alpha.

by Kevin bagus saputra

---

## 1. Cara Install Dependencies agar library tersedia

Pastikan Python 3.10+ sudah terinstall.  
Install dependencies menggunakan `pip`:

```bash
pip install -r requirements.txt
```

## 2. cara menjalankan script untuk generate model

```bash
python polynomial_regression_pipeline.py
```

## 3 cara menggunakan model terbaik

```bash
python coba_model.py
```

## 4 struktur folder

```bash
UTS/
│
├─ polynomial_regression_pipeline.py      # Script utama pipeline
├─ coba_model.py                          # Script untuk mencoba prediksi baru
├─ requirements.txt                       # Library dependencies
│
├─ polynomial_regression_project_artifacts/
│   ├─ synthetic_property_data.csv        # Dataset (synthetic)
│   ├─ best_model.pkl                     # Model terbaik (pickle)
│   ├─ best_poly.pkl                      # Polynomial transformer terbaik (pickle)
│   ├─ scaler.pkl                          # Scaler untuk preprocessing
│   ├─ model_results_summary.csv           # Summary hasil training semua model
│   ├─ final_predictions_with_CI.csv       # Prediksi baru dengan confidence interval
│   ├─ feature_importance.csv              # Koefisien Ridge & Lasso
│   ├─ report.md                           # Laporan akhir
│   ├─ eda_plots/                          # Hasil plot EDA (histogram, scatter, box, heatmap)
│   └─ evaluation_plots/                   # Plot evaluasi model (learning curve, regularization, feature importance)
```
