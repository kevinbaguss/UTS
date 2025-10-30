import joblib
import pandas as pd
import numpy as np

# --- 1. Load Artefak ---
try:
    # Memuat artefak model yang telah dilatih menggunakan joblib
    scaler = joblib.load('polynomial_regression_project_artifacts/scaler.pkl')
    poly = joblib.load('polynomial_regression_project_artifacts/best_poly.pkl')
    model = joblib.load('polynomial_regression_project_artifacts/best_model.pkl')
    print("✅ Artefak model berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: File artefak tidak ditemukan. Pastikan Anda telah menjalankan pipeline dan file 'scaler.pkl', 'best_poly.pkl', dan 'best_model.pkl' ada di folder 'polynomial_regression_project_artifacts'.")
    exit()

# --- 2. Input Data dari Pengguna ---
print("\n🏡 Masukkan detail properti untuk mendapatkan prediksi harga:")

try:
    # Input interaktif dari pengguna. Nilai dikonversi ke float/int.
    luas_tanah = float(input("Masukkan Luas Tanah (m2): "))
    luas_bangunan = float(input("Masukkan Luas Bangunan (m2): "))
    jumlah_kamar = int(input("Masukkan Jumlah Kamar: "))
    umur_bangunan = float(input("Masukkan Umur Bangunan (tahun): "))
    jarak_pusat = float(input("Masukkan Jarak ke Pusat Kota (km): "))

except ValueError:
    print("❌ ERROR: Input harus berupa angka. Silakan coba lagi.")
    exit()


# --- 3. Pembentukan Data Baru ---
# Data harus sesuai dengan urutan dan nama fitur saat pelatihan!
new_data_input = pd.DataFrame({
    'Luas_Tanah_m2': [luas_tanah],
    'Luas_Bangunan_m2': [luas_bangunan],
    'Jumlah_Kamar': [jumlah_kamar],
    'Umur_Bangunan_th': [umur_bangunan],
    'Jarak_ke_Pusat_km': [jarak_pusat]
})

# --- 4. Transformasi & Prediksi ---
print("\n⏳ Melakukan transformasi data dan prediksi...")

# 1. Scaling data input
X_scaled = scaler.transform(new_data_input)

# 2. Transformasi polinomial
X_poly = poly.transform(X_scaled)

# 3. Prediksi menggunakan model terbaik
prediction = model.predict(X_poly)

# --- 5. Tampilkan Hasil ---
predicted_price = prediction[0]

print("\n=======================================================")
print("  💰 Prediksi Harga Properti (berdasarkan model terbaik)")
print(f"  Harga: **Rp {predicted_price:,.2f} Juta**")
print("=======================================================")