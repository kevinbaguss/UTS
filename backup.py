import os
import numpy as np
import pandas as pd
from math import sqrt
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- Helpers ----------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_pred), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved CSV -> {path}")

# ---------- Config ----------
OUT_DIR = "./polynomial_regression_project_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_CSV = os.path.join(OUT_DIR, "synthetic_property_data.csv")
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# ---------- 1) DATA GENERATION (LOGIS) ----------
# ---------------------------------------------------------------------
if os.path.exists(DATA_CSV):
    print("Loading existing dataset:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
else:
    print("Generating synthetic dataset (n=300) dengan formula logis...")
    np.random.seed(RANDOM_STATE)
    n = 300
    
    # Batas Fitur
    RANGES = {
        "Luas_Tanah_m2": (50, 500),
        "Luas_Bangunan_m2": (30, 400),
        "Jumlah_Kamar": (1, 5),
        "Umur_Bangunan_th": (0, 30),
        "Jarak_ke_Pusat_km": (1, 20)
    }
    PRICE_MIN = 200000000
    PRICE_MAX = 5000000000
    
    # Generate Fitur
    luas_tanah = np.random.uniform(RANGES["Luas_Tanah_m2"][0], RANGES["Luas_Tanah_m2"][1], n)
    luas_bangunan = np.random.uniform(RANGES["Luas_Bangunan_m2"][0], RANGES["Luas_Bangunan_m2"][1], n)
    jml_kamar = np.random.randint(RANGES["Jumlah_Kamar"][0], RANGES["Jumlah_Kamar"][1] + 1, n)
    umur = np.random.uniform(RANGES["Umur_Bangunan_th"][0], RANGES["Umur_Bangunan_th"][1], n)
    jarak = np.random.uniform(RANGES["Jarak_ke_Pusat_km"][0], RANGES["Jarak_ke_Pusat_km"][1], n)

    
    # Koefisien untuk Formula Logis (skala juta rupiah)
    K_LT = 5e6    # 5 juta m2 tanah
    K_LB = 7e6    # 7 juta per m2 bangunan
    K_JK = 300e6  # 150 juta per kamar


    # Formula Harga Properti (Logis & Non-Linear)
    price = (
        200e6                                                # Base price
        + K_LB * luas_bangunan                             # Nilai dari Luas Bangunan
        + K_LT * luas_tanah                                # Nilai dari Luas Tanah
        + K_JK * jml_kamar                                 # Nilai Jumlah Kamar

        # Interaksi & Suku Polinomial
        + 0.5e6 * (luas_bangunan * jml_kamar)                # Interaksi: LB besar + Kamar banyak -> premium
        - 2e5 * (luas_tanah * jarak)                      # Interaksi: LT besar JAUH -> diskon nilai tanah

        # Faktor Non-Linear: Depresiasi (Log) dan Lokasi (Inverse)
        - 8e6 * np.log(umur + 1)                          # Depresiasi: Penurunan tajam di awal, melandai
        + 400e6 / (jarak + 0.5)                            # Premium Lokasi: Sangat tinggi dekat pusat, menurun drastis
    )

    # Tambahkan Noise (sekitar 15% dari harga) dan Clip
    price = np.maximum(price, 5e6)
    noise = np.random.normal(0, 0.15 * np.abs(price),n)
    price = np.clip(price + noise, PRICE_MIN, PRICE_MAX)

    df = pd.DataFrame({
        "Luas_Tanah_m2": luas_tanah,
        "Luas_Bangunan_m2": luas_bangunan,
        "Jumlah_Kamar": jml_kamar,
        "Umur_Bangunan_th": umur,
        "Jarak_ke_Pusat_km": jarak,
        "Harga_Juta": price
    })
    save_csv(df, DATA_CSV)

# ---------------------------------------------------------------------

# ---------- 2) EDA (summary & simple) ----------
print("\n=== Dataset summary ===")
display_df = df.describe().T
print(display_df)

# Uncomment below lines to produce plots (if running locally)
# for col in df.columns:
#     plt.figure(figsize=(6,3))
#     plt.hist(df[col], bins=25)
#     plt.title(f"Histogram: {col}")
#     plt.xlabel(col); plt.ylabel("Count")
#     plt.tight_layout()
#     plt.show()

# ---------- 3) Preprocessing ----------
features = ["Luas_Tanah_m2", "Luas_Bangunan_m2", "Jumlah_Kamar", "Umur_Bangunan_th", "Jarak_ke_Pusat_km"]
X = df[features].copy()
y = df["Harga_Juta"].copy()

# handle missing values if any
if df.isnull().sum().sum() > 0:
    print("Filling missing values with median...")
    df = df.fillna(df.median())
    X = df[features]; y = df["Harga_Juta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
print("Scaler saved.")

# ---------- 4) Model training & evaluation (degrees 1..5) ----------
degrees = [1, 2, 3, 4, 5]
alphas_small = [0.1, 1, 10]
models = {}
results = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Xtr_poly = poly.fit_transform(X_train_s)
    Xte_poly = poly.transform(X_test_s)
    n_feat = Xtr_poly.shape[1]

    # Linear
    lr = LinearRegression().fit(Xtr_poly, y_train)
    models[f"deg{deg}_linear"] = {"poly": poly, "model": lr}

    def eval_and_store(m, model_name):
        ytr = m.predict(Xtr_poly); yte = m.predict(Xte_poly)
        res = {
            "degree": deg,
            "model": model_name,
            "n_features": n_feat,
            "train_r2": r2_score(y_train, ytr),
            "test_r2": r2_score(y_test, yte),
            "train_mse": mean_squared_error(y_train, ytr),
            "test_mse": mean_squared_error(y_test, yte),
            "train_rmse": sqrt(mean_squared_error(y_train, ytr)),
            "test_rmse": sqrt(mean_squared_error(y_test, yte)),
            "train_mae": mean_absolute_error(y_train, ytr),
            "test_mae": mean_absolute_error(y_test, yte),
            "train_mape": mape(y_train, ytr),
            "test_mape": mape(y_test, yte)
        }
        results.append(res)

    eval_and_store(lr, "Linear")

    # Ridge & Lasso
    for a in alphas_small:
        r = Ridge(alpha=a, max_iter=10000).fit(Xtr_poly, y_train)
        l = Lasso(alpha=a, max_iter=10000).fit(Xtr_poly, y_train)
        models[f"deg{deg}_ridge_{a}"] = {"poly": poly, "model": r}
        models[f"deg{deg}_lasso_{a}"] = {"poly": poly, "model": l}
        eval_and_store(r, f"Ridge_{a}")
        eval_and_store(l, f"Lasso_{a}")

# Save results table
results_df = pd.DataFrame(results).sort_values(["degree","model"])
results_csv = os.path.join(OUT_DIR, "model_results_summary.csv")
results_df.to_csv(results_csv, index=False)
print(f"Model results saved -> {results_csv}")

# Print top few by test_r2
top_test = results_df.sort_values("test_r2", ascending=False).head(8)
print("\nTop models by test R2:")
print(top_test[["degree","model","n_features","test_r2","test_rmse","test_mae"]])

# ---------- 5) Regularization scan for degree 3 ----------
poly3 = PolynomialFeatures(degree=3, include_bias=False)
Xtr3 = poly3.fit_transform(X_train_s)
Xte3 = poly3.transform(X_test_s)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = []
lasso_scores = []
for a in alphas:
    r = Ridge(alpha=a, max_iter=10000).fit(Xtr3, y_train)
    l = Lasso(alpha=a, max_iter=10000).fit(Xtr3, y_train)
    ridge_scores.append(r2_score(y_test, r.predict(Xte3)))
    lasso_scores.append(r2_score(y_test, l.predict(Xte3)))

print("\nRegularization scan (degree=3) results (alpha -> test R2):")
print("Ridge:", list(zip(alphas, ridge_scores)))
print("Lasso:", list(zip(alphas, lasso_scores)))

best_ridge_alpha = alphas[int(np.argmax(ridge_scores))]
best_lasso_alpha = alphas[int(np.argmax(lasso_scores))]
print("Best Ridge alpha:", best_ridge_alpha)
print("Best Lasso alpha:", best_lasso_alpha)

# ---------- 6) Cross-validation model selection ----------
candidates = []
for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Xpoly_all = poly.fit_transform(scaler.transform(X))
    lr = LinearRegression()
    sc = cross_val_score(lr, Xpoly_all, y, cv=5, scoring='r2')
    candidates.append({"model":"Linear","degree":deg,"cv_r2_mean":sc.mean(),"cv_r2_std":sc.std()})

# Add Ridge and Lasso degree 3 with best alphas
poly3_all = PolynomialFeatures(degree=3, include_bias=False)
Xpoly3_all = poly3_all.fit_transform(scaler.transform(X))
ridge_cv = Ridge(alpha=best_ridge_alpha, max_iter=10000)
lasso_cv = Lasso(alpha=best_lasso_alpha, max_iter=10000)
candidates.append({"model":"Ridge","degree":3,"cv_r2_mean":cross_val_score(ridge_cv,Xpoly3_all,y,cv=5,scoring='r2').mean(),"cv_r2_std":0.0})
candidates.append({"model":"Lasso","degree":3,"cv_r2_mean":cross_val_score(lasso_cv,Xpoly3_all,y,cv=5,scoring='r2').mean(),"cv_r2_std":0.0})

candidates_df = pd.DataFrame(candidates).sort_values("cv_r2_mean", ascending=False)
candidates_path = os.path.join(OUT_DIR, "cv_candidates.csv")
candidates_df.to_csv(candidates_path, index=False)
print("\nCross-validation candidates saved ->", candidates_path)
print(candidates_df)

best = candidates_df.iloc[0]
print("\nSelected best candidate by CV:", best.to_dict())

# ---------- 7) Train final model (selected) on train set & save artifacts ----------
if best['model'] == "Linear":
    deg_best = int(best['degree'])
    poly_best = models[f"deg{deg_best}_linear"]["poly"]
    model_best = models[f"deg{deg_best}_linear"]["model"]
else:
    deg_best = int(best['degree'])
    poly_best = PolynomialFeatures(degree=deg_best, include_bias=False)
    if best['model'] == "Ridge":
        model_best = Ridge(alpha=best_ridge_alpha, max_iter=10000).fit(poly_best.fit_transform(X_train_s), y_train)
    else:
        model_best = Lasso(alpha=best_lasso_alpha, max_iter=10000).fit(poly_best.fit_transform(X_train_s), y_train)

# Save
joblib.dump(model_best, os.path.join(OUT_DIR, "best_model.pkl"))
joblib.dump(poly_best, os.path.join(OUT_DIR, "best_poly.pkl"))
print("Saved best model and poly transformer to", OUT_DIR)

# ---------- 8) Final prediction on 5 new samples ----------
new_samples = pd.DataFrame({
    "Luas_Tanah_m2":[60,120,250,400,320],
    "Luas_Bangunan_m2":[40,100,200,350,280],
    "Jumlah_Kamar":[1,2,3,4,5],
    "Umur_Bangunan_th":[1,5,10,20,2],
    "Jarak_ke_Pusat_km":[2,5,10,15,8]
})
Xnew_s = scaler.transform(new_samples)
Xnew_p = poly_best.transform(Xnew_s)
ynew = model_best.predict(Xnew_p)
new_samples["Predicted_Harga_juta"] = ynew

pred_path = os.path.join(OUT_DIR, "predictions_5_new.csv")
save_csv(new_samples, pred_path)

# ---------- 9) Short report ----------
report_txt = os.path.join(OUT_DIR, "report.txt")
with open(report_txt, "w") as f:
    f.write("Polynomial Regression Project - short report\n")
    f.write("\nDataset: " + DATA_CSV + "\n")
    f.write("\nTop models by test R2 (sample):\n")
    f.write(top_test.to_string())
    f.write("\n\nSelected best CV candidate:\n")
    f.write(str(best.to_dict()))
print("Report saved ->", report_txt)

print("\nAll artifacts saved in folder:", OUT_DIR)