# Polynomial Regression Project - Report

## 1. Executive Summary

- Dataset: 300 sample properties
- Best Polynomial Degree: 2
- Best CV R2 (Linear Model): 0.8106
- Best Regularization (Degree=2): Ridge alpha=10, R2=0.7986

## 2. Insights from EDA

- Fitur `Luas_Bangunan_m2` dan `Luas_Tanah_m2` memiliki korelasi positif kuat terhadap harga.
- `Jarak_ke_Pusat_km` berpengaruh negatif terhadap harga.
- Terdapat beberapa outlier pada `Harga_Juta` dan `Luas_Tanah_m2`.

## 3. Perbandingan Performa Model

```bash

|     | degree | model       |  test_r2 |   test_rmse |
| --: | -----: | :---------- | -------: | ----------: |
|   0 |      1 | Linear      | 0.820541 | 4.62502e+08 |
|   2 |      1 | Lasso_0.001 | 0.820541 | 4.62502e+08 |
|   4 |      1 | Lasso_0.01  | 0.820541 | 4.62502e+08 |
|   6 |      1 | Lasso_0.1   | 0.820541 | 4.62502e+08 |
|   8 |      1 | Lasso_1     | 0.820541 | 4.62502e+08 |

```

## 4. Rekomendasi

- Polynomial Degree terbaik: 2
- Regularization Method terbaik: Ridge alpha=10

## 5. Limitations

- Model hanya menggunakan fitur dasar, belum termasuk faktor lokasi micro, fasilitas, kondisi bangunan.
- Model rentan terhadap outlier besar di harga.
- Prediksi CI berbasis residual std, belum probabilistik penuh.

## 6. Suggested Improvements

- Tambahkan fitur kualitatif seperti fasilitas, lingkungan.
- Gunakan cross-validation lebih kompleks atau bootstrap.
