

import os
import numpy as np
import pandas as pd
from math import sqrt
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------- Helpers -----------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved CSV -> {path}")

def create_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def plot_save_hist(df, out_dir):
    for col in df.columns:
        plt.figure(figsize=(6,3))
        plt.hist(df[col], bins=25, color='skyblue', edgecolor='black')
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{col}.png"))
        plt.close()

def plot_save_scatter(df, features, out_dir):
    for col in features:
        plt.figure(figsize=(5,4))
        plt.scatter(df[col], df["Harga_Juta"], alpha=0.6, color='coral')
        plt.xlabel(col)
        plt.ylabel("Harga_Juta")
        plt.title(f"{col} vs Harga_Juta")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scatter_{col}.png"))
        plt.close()

def plot_save_heatmap(df, out_dir):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.close()

def plot_save_box(df, features, out_dir):
    for col in features:
        plt.figure(figsize=(5,3))
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"Outlier Check: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"boxplot_{col}.png"))
        plt.close()

def predict_new_data(model, poly, scaler, new_data, y_test, X_test_s):
    Xs = scaler.transform(new_data)
    Xp = poly.transform(Xs)
    preds = model.predict(Xp)
    resid_std = np.std(y_test - model.predict(poly.transform(X_test_s)))
    lower = preds - 1.96 * resid_std
    upper = preds + 1.96 * resid_std
    result = new_data.copy()
    result["Predicted_Harga"] = preds
    result["Lower_CI"] = lower
    result["Upper_CI"] = upper
    return result

# ----------------- Config -----------------
OUT_DIR = "./polynomial_regression_project_artifacts"
EDA_DIR = os.path.join(OUT_DIR, "eda_plots")
EVAL_DIR = os.path.join(OUT_DIR, "evaluation_plots")
DATA_CSV = os.path.join(OUT_DIR, "synthetic_property_data.csv")
RANDOM_STATE = 42

create_dirs([EDA_DIR, EVAL_DIR])

# ----------------- 1) Data Generation -----------------
if os.path.exists(DATA_CSV):
    print("Loading existing dataset:", DATA_CSV)
    df = pd.read_csv(DATA_CSV)
else:
    print("Generating synthetic dataset (n=300)...")
    np.random.seed(RANDOM_STATE)
    n = 300
    RANGES = {
        "Luas_Tanah_m2": (50, 500),
        "Luas_Bangunan_m2": (30, 400),
        "Jumlah_Kamar": (1, 5),
        "Umur_Bangunan_th": (0, 30),
        "Jarak_ke_Pusat_km": (1, 20)
    }
    PRICE_MIN, PRICE_MAX = 200_000_000, 5_000_000_000

    luas_tanah = np.random.uniform(*RANGES["Luas_Tanah_m2"], n)
    luas_bangunan = np.random.uniform(*RANGES["Luas_Bangunan_m2"], n)
    jml_kamar = np.random.randint(*RANGES["Jumlah_Kamar"], n)
    umur = np.random.uniform(*RANGES["Umur_Bangunan_th"], n)
    jarak = np.random.uniform(*RANGES["Jarak_ke_Pusat_km"], n)

    K_LT, K_LB, K_JK = 5e6, 7e6, 300e6
    price = (
        200e6
        + K_LB * luas_bangunan
        + K_LT * luas_tanah
        + K_JK * jml_kamar
        + 0.5e6 * (luas_bangunan * jml_kamar)
        - 2e5 * (luas_tanah * jarak)
        - 8e6 * np.log(umur + 1)
        + 400e6 / (jarak + 0.5)
    )
    price = np.maximum(price, 5e6)
    noise = np.random.normal(0, 0.15 * np.abs(price), n)
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

# ----------------- 2) EDA -----------------
features = ["Luas_Tanah_m2","Luas_Bangunan_m2","Jumlah_Kamar","Umur_Bangunan_th","Jarak_ke_Pusat_km"]
print("\n=== Dataset Summary ===")
print(df.describe().T)

plot_save_hist(df, EDA_DIR)
plot_save_scatter(df, features, EDA_DIR)
plot_save_heatmap(df, EDA_DIR)
plot_save_box(df, features, EDA_DIR)
print(f"EDA plots saved in {EDA_DIR}")

# ----------------- 3) Preprocessing -----------------
X = df[features].copy()
y = df["Harga_Juta"].copy()
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median())
    X = df[features]; y = df["Harga_Juta"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT_DIR,"scaler.pkl"))

# ----------------- 4) Model Training & Evaluation -----------------
degrees = [1,2,3,4,5]
alphas_extended = [0.001,0.01,0.1,1,10,100]
models, results = {}, []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Xtr_poly, Xte_poly = poly.fit_transform(X_train_s), poly.transform(X_test_s)
    n_feat = Xtr_poly.shape[1]

    # Linear Regression
    lr = LinearRegression().fit(Xtr_poly, y_train)
    models[f"deg{deg}_linear"] = {"poly": poly, "model": lr}

    def eval_store(m, name):
        ytr, yte = m.predict(Xtr_poly), m.predict(Xte_poly)
        results.append({
            "degree": deg, "model": name, "n_features": n_feat,
            "train_r2": r2_score(y_train, ytr),
            "test_r2": r2_score(y_test, yte),
            "train_rmse": sqrt(mean_squared_error(y_train, ytr)),
            "test_rmse": sqrt(mean_squared_error(y_test, yte)),
            "train_mae": mean_absolute_error(y_train, ytr),
            "test_mae": mean_absolute_error(y_test, yte),
            "train_mape": mape(y_train, ytr),
            "test_mape": mape(y_test, yte)
        })
    eval_store(lr, "Linear")

    # Ridge & Lasso
    for a in alphas_extended:
        ridge = Ridge(alpha=a, max_iter=10000).fit(Xtr_poly, y_train)
        lasso = Lasso(alpha=a, max_iter=10000).fit(Xtr_poly, y_train)
        models[f"deg{deg}_ridge_{a}"] = {"poly": poly, "model": ridge}
        models[f"deg{deg}_lasso_{a}"] = {"poly": poly, "model": lasso}
        eval_store(ridge, f"Ridge_{a}")
        eval_store(lasso, f"Lasso_{a}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR,"model_results_summary.csv"), index=False)

# Learning Curve
plt.figure(figsize=(6,4))
deg_mean = results_df.groupby("degree")[["train_r2","test_r2"]].mean().reset_index()
plt.plot(deg_mean["degree"], deg_mean["train_r2"], marker='o', label="Train R²")
plt.plot(deg_mean["degree"], deg_mean["test_r2"], marker='o', label="Test R²")
plt.xlabel("Polynomial Degree"); plt.ylabel("R² Score")
plt.title("Learning Curve: Train vs Test R²"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,"learning_curve_r2.png"))
plt.close()

# ----------------- 5) Cross-validation -----------------
candidates = []
for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Xpoly_all = poly.fit_transform(scaler.transform(X))
    lr = LinearRegression()
    sc = cross_val_score(lr, Xpoly_all, y, cv=5, scoring='r2')
    candidates.append({"model":"Linear","degree":deg,"cv_r2_mean":sc.mean(),"cv_r2_std":sc.std()})
candidates_df = pd.DataFrame(candidates).sort_values("cv_r2_mean", ascending=False)
best = candidates_df.iloc[0]
deg_best = int(best["degree"])
poly_best = models[f"deg{deg_best}_linear"]["poly"]
model_best = models[f"deg{deg_best}_linear"]["model"]
joblib.dump(model_best, os.path.join(OUT_DIR,"best_model.pkl"))
joblib.dump(poly_best, os.path.join(OUT_DIR,"best_poly.pkl"))

# ----------------- 6) Regularization Analysis -----------------
deg_target = 2
poly_target = PolynomialFeatures(degree=deg_target, include_bias=False)
Xtr_poly_t, Xte_poly_t = poly_target.fit_transform(X_train_s), poly_target.transform(X_test_s)

ridge_scores, lasso_scores = [], []
for a in alphas_extended:
    ridge = Ridge(alpha=a, max_iter=10000).fit(Xtr_poly_t, y_train)
    lasso = Lasso(alpha=a, max_iter=10000).fit(Xtr_poly_t, y_train)
    ridge_scores.append((a, r2_score(y_test, ridge.predict(Xte_poly_t))))
    lasso_scores.append((a, r2_score(y_test, lasso.predict(Xte_poly_t))))

plt.figure(figsize=(6,4))
plt.plot([a for a,_ in ridge_scores],[s for _,s in ridge_scores], marker='o', label='Ridge')
plt.plot([a for a,_ in lasso_scores],[s for _,s in lasso_scores], marker='o', label='Lasso')
plt.xscale('log'); plt.xlabel("Alpha"); plt.ylabel("R² Score")
plt.title(f"Ridge vs Lasso (Degree={deg_target})"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,"ridge_lasso_r2_vs_alpha.png"))
plt.close()

best_ridge = max(ridge_scores,key=lambda x:x[1])
best_lasso = max(lasso_scores,key=lambda x:x[1])
ridge_best = Ridge(alpha=best_ridge[0]).fit(Xtr_poly_t, y_train)
lasso_best = Lasso(alpha=best_lasso[0]).fit(Xtr_poly_t, y_train)

coef_df = pd.DataFrame({
    "Feature": poly_target.get_feature_names_out(features),
    "Ridge_coef": ridge_best.coef_,
    "Lasso_coef": lasso_best.coef_
})
coef_df["Lasso_zeroed"] = coef_df["Lasso_coef"]==0
coef_df.to_csv(os.path.join(OUT_DIR,"feature_importance.csv"), index=False)

plt.figure(figsize=(8,5))
coef_sorted = coef_df.sort_values("Ridge_coef", key=lambda x: np.abs(x), ascending=False).head(15)
plt.barh(coef_sorted["Feature"], coef_sorted["Ridge_coef"], color='steelblue', label="Ridge")
plt.barh(coef_sorted["Feature"], coef_sorted["Lasso_coef"], color='coral', alpha=0.6, label="Lasso")
plt.gca().invert_yaxis(); plt.xlabel("Coefficient Value"); plt.title("Feature Importance Ridge vs Lasso")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR,"feature_importance_ridge_lasso.png"))
plt.close()

# ----------------- 7) Final Predictions -----------------
new_samples = pd.DataFrame({
    "Luas_Tanah_m2":[60,120,250,400,320],
    "Luas_Bangunan_m2":[40,100,200,350,280],
    "Jumlah_Kamar":[1,2,3,4,5],
    "Umur_Bangunan_th":[1,5,10,20,2],
    "Jarak_ke_Pusat_km":[2,5,10,15,8]
})
pred_result = predict_new_data(model_best, poly_best, scaler, new_samples, y_test, X_test_s)
save_csv(pred_result, os.path.join(OUT_DIR,"final_predictions_with_CI.csv"))

# ----------------- 8) Report & Insights -----------------
report_path = os.path.join(OUT_DIR,"report.md")
with open(report_path,"w") as f:
    f.write("# Polynomial Regression Project - Report\n\n")
    f.write("## 1. Executive Summary\n")
    f.write(f"- Dataset: {len(df)} sample properties\n")
    f.write(f"- Best Polynomial Degree: {deg_best}\n")
    f.write(f"- Best CV R² (Linear Model): {best['cv_r2_mean']:.4f}\n")
    f.write(f"- Best Regularization (Degree={deg_target}): Ridge alpha={best_ridge[0]}, R²={best_ridge[1]:.4f}\n\n")
    
    f.write("## 2. Insights from EDA\n")
    f.write("- Fitur `Luas_Bangunan_m2` dan `Luas_Tanah_m2` memiliki korelasi positif kuat terhadap harga.\n")
    f.write("- `Jarak_ke_Pusat_km` berpengaruh negatif terhadap harga.\n")
    f.write("- Terdapat beberapa outlier pada `Harga_Juta` dan `Luas_Tanah_m2`.\n\n")
    
    f.write("## 3. Perbandingan Performa Model\n")
    top_models = results_df.sort_values('test_r2', ascending=False).head(5)
    f.write(top_models[['degree','model','test_r2','test_rmse']].to_markdown())
    f.write("\n\n")
    
    f.write("## 4. Rekomendasi\n")
    f.write(f"- Polynomial Degree terbaik: {deg_best}\n")
    f.write(f"- Regularization Method terbaik: Ridge alpha={best_ridge[0]}\n\n")
    
    f.write("## 5. Limitations\n")
    f.write("- Model hanya menggunakan fitur dasar, belum termasuk faktor lokasi micro, fasilitas, kondisi bangunan.\n")
    f.write("- Model rentan terhadap outlier besar di harga.\n")
    f.write("- Prediksi CI berbasis residual std, belum probabilistik penuh.\n\n")
    
    f.write("## 6. Suggested Improvements\n")
    f.write("- Tambahkan fitur kualitatif seperti fasilitas, lingkungan.\n")
    f.write("- Gunakan cross-validation lebih kompleks atau bootstrap.\n")
    f.write("- Eksperimen dengan model tree-based (RandomForest, XGBoost) untuk non-linear yang lebih kompleks.\n")

print(f"Report saved: {report_path}")

# ----------------- 9) README & requirements.txt -----------------
readme_path = os.path.join(OUT_DIR,"README.md")
with open(readme_path,"w") as f:
    f.write("# Polynomial Regression Project\n\n")
    f.write("## Dependencies\n```bash\npip install -r requirements.txt\n```\n\n")
    f.write("## Run Script\n```bash\npython polynomial_regression.py\n```\n\n")
    f.write("## Project Structure\n")
    f.write("- polynomial_regression_project_artifacts/\n")
    f.write("  - synthetic_property_data.csv\n")
    f.write("  - best_model.pkl\n")
    f.write("  - best_poly.pkl\n")
    f.write("  - model_results_summary.csv\n")
    f.write("  - report.md\n")
    f.write("  - eda_plots/\n")
    f.write("  - evaluation_plots/\n")

req_path = os.path.join(OUT_DIR,"requirements.txt")
with open(req_path,"w") as f:
    f.write("numpy\npandas\nscikit-learn\nmatplotlib\nseaborn\njoblib\n")

print(f"README and requirements.txt saved in {OUT_DIR}")
