# src/03_modeling.py
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay

# -------------------------------------------------------------------
# I/O
# -------------------------------------------------------------------
DERIVED = Path("data/derived")
PLOTS = DERIVED / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

lon = pd.read_csv(DERIVED / "species_longevity.csv")
clim = pd.read_csv(DERIVED / "species_climate.csv")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def plant_group(species: str) -> str:
    """Very simple genus-based grouping to illustrate between-group differences."""
    if not isinstance(species, str) or not species.strip():
        return "Unknown"
    genus = species.split()[0].lower()
    conifer_genera = {"pinus", "picea", "abies", "larix", "juniperus", "cedrus", "tsuga", "pseudotsuga", "sequoia"}
    return "Conifer" if genus in conifer_genera else "Angiosperm"

def tidy_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def kfold(n_obs: int) -> KFold:
    k = min(5, max(2, n_obs // 3))
    return KFold(n_splits=k, shuffle=True, random_state=42)

def evaluate_cv(model, X, y, scoring="r2"):
    # scikit-learn uses "neg_mean_squared_error" for MSE; convert later
    if scoring == "mse":
        scores = cross_val_score(model, X, y, cv=kfold(len(y)), scoring="neg_mean_squared_error")
        return -scores.mean()
    elif scoring == "mae":
        scores = cross_val_score(model, X, y, cv=kfold(len(y)), scoring="neg_mean_absolute_error")
        return -scores.mean()
    else:
        scores = cross_val_score(model, X, y, cv=kfold(len(y)), scoring="r2")
        return scores.mean()

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")

# -------------------------------------------------------------------
# Prepare data
# -------------------------------------------------------------------
lon["plant_group"] = lon["species"].map(plant_group)
dat = lon.merge(clim, on="species", how="left")

# numeric features
feature_cols = ["bio1", "bio12", "bio4"]
dat = tidy_numeric(dat, feature_cols + ["max_series_years"])
dat = dat.dropna(subset=["max_series_years"])

# design matrices
X_base = pd.get_dummies(dat[feature_cols + ["plant_group"]], drop_first=True)
X_base = tidy_numeric(X_base, X_base.columns).astype("float64")
y = dat["max_series_years"].astype("float64")

mask = X_base.notna().all(axis=1) & y.notna()
X = X_base.loc[mask]
y = y.loc[mask]
n = len(y)

print(f"Rows available for modeling: n={n}, p={X.shape[1]}")

if n < 8:
    warnings.warn("Very small sample size; CV and diagnostics may be unstable.", RuntimeWarning)


# OLS with interactions & curvature (bio1^2 and bio1 × group)
X_poly = X.copy()
if "bio1" in X_poly.columns:
    X_poly["bio1_sq"] = X_poly["bio1"] ** 2
if "plant_group_Conifer" in X_poly.columns and "bio1" in X_poly.columns:
    X_poly["bio1_x_conifer"] = X_poly["bio1"] * X_poly["plant_group_Conifer"]

X_poly = sm.add_constant(X_poly, has_constant="add")
ols2_res = sm.OLS(y, X_poly, missing="drop").fit(cov_type="HC3")
with open(DERIVED / "model_ols_interact_summary.txt", "w") as f:
    f.write(ols2_res.summary().as_text())
print("Saved: data/derived/model_ols_interact_summary.txt")

# MODEL : Ridge regression (CV over alphas)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)
alphas = np.logspace(-3, 3, 33)
ridge = RidgeCV(alphas=alphas, cv=kfold(n)).fit(X_scaled, y.values)

# Ridge CV curve (manual CV for visualization)
cv_mse = []
for a in alphas:
    model = RidgeCV(alphas=[a], cv=kfold(n))
    # emulate CV curve via cross_val_score on a fixed alpha
    # we use a plain Ridge regressor via LinearRegression on scaled X; simpler is to fit Ridge(a)
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=a)
    mse = -cross_val_score(m, X_scaled, y.values, cv=kfold(n), scoring="neg_mean_squared_error").mean()
    cv_mse.append(mse)

plt.figure()
plt.semilogx(alphas, cv_mse, marker="o")
plt.xlabel("Ridge alpha (log scale)")
plt.ylabel("CV MSE")
plt.title("RidgeCV: cross-validated MSE")
savefig(PLOTS / "ridge_cv_curve.png")

ridge_coef = pd.Series(ridge.coef_, index=X.columns)
ridge_coef.sort_values(ascending=False).to_csv(DERIVED / "ridge_coefficients.csv")
print("Saved: data/derived/ridge_coefficients.csv")

#  Lasso regression (CV)
lasso = LassoCV(alphas=None, cv=kfold(n), max_iter=20000, random_state=42)
lasso.fit(X_scaled, y.values)

# Lasso path (approximate via trained alphas_)
if hasattr(lasso, "alphas_"):
    plt.figure()
    plt.semilogx(lasso.alphas_, lasso.mse_path_.mean(axis=1), marker="o")
    plt.xlabel("Lasso alpha (log scale)")
    plt.ylabel("CV MSE")
    plt.title("LassoCV: cross-validated MSE")
    savefig(PLOTS / "lasso_cv_curve.png")

lasso_coef = pd.Series(lasso.coef_, index=X.columns)
lasso_coef.sort_values(ascending=False).to_csv(DERIVED / "lasso_coefficients.csv")
print("Saved: data/derived/lasso_coefficients.csv")


# MODEL 5: Random Forest 
rf = RandomForestRegressor(
    n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf.fit(X, y)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure()
plt.barh(rf_importance.index, rf_importance.values)
plt.xlabel("Feature importance")
plt.title("Random Forest: feature importances")
savefig(PLOTS / "rf_feature_importance.png")

# Compare

def cv_metrics(estimator, X_eval, y_eval, scaled=False):
    Xe = X_eval if not scaled else scaler.fit_transform(X_eval.values)
    if scaled:  # re-fit scaler for fair CV each time
        X_for_cv = Xe
    else:
        X_for_cv = X_eval
    r2 = evaluate_cv(estimator, X_for_cv, y_eval, scoring="r2")
    mse = evaluate_cv(estimator, X_for_cv, y_eval, scoring="mse")
    mae = evaluate_cv(estimator, X_for_cv, y_eval, scoring="mae")
    return r2, mse, mae

models = []
# OLS (use LinearRegression on X for CV, since statsmodels lacks cross_val_score API)
lr = LinearRegression()
models.append(("OLS", lr, False))
# OLS with interactions/curvature
cols_poly = [c for c in X_poly.columns if c != "const"]
lr2 = LinearRegression()
models.append(("OLS+poly", lr2, False))
# Ridge/Lasso (scaled)
models.append(("RidgeCV", RidgeCV(alphas=alphas, cv=kfold(n)), True))
models.append(("LassoCV", LassoCV(cv=kfold(n), max_iter=20000, random_state=42), True))
# Random Forest
models.append(("RandomForest", rf, False))

rows = []
for name, est, needs_scaled in models:
    if name == "OLS+poly":
        Xe = X_poly[cols_poly]
    else:
        Xe = X
    try:
        r2_cv, mse_cv, mae_cv = cv_metrics(est, Xe, y, scaled=needs_scaled)
    except Exception as e:
        warnings.warn(f"CV failed for {name}: {e}")
        r2_cv, mse_cv, mae_cv = (np.nan, np.nan, np.nan)
    rows.append({"model": name, "r2_cv": r2_cv, "rmse_cv": np.sqrt(mse_cv) if pd.notna(mse_cv) else np.nan, "mae_cv": mae_cv})

cmp_df = pd.DataFrame(rows).sort_values("r2_cv", ascending=False)
cmp_df.to_csv(DERIVED / "model_comparison.csv", index=False)
print("Saved: data/derived/model_comparison.csv")
print(cmp_df)


# 1) Scatter: longevity vs climate, colored by group, with simple OLS line
plt.figure()
groups = dat.loc[mask, "plant_group"].values
x = X["bio1"] if "bio1" in X.columns else None
if x is not None:
    for g in np.unique(groups):
        sel = groups == g
        plt.scatter(x[sel], y[sel], label=g, s=40, alpha=0.8)
        # fit simple line per group if enough points
        if sel.sum() >= 3:
            lr_g = LinearRegression().fit(x[sel].to_numpy().reshape(-1, 1), y[sel].to_numpy())
            xs = np.linspace(x.min(), x.max(), 100)
            ys = lr_g.predict(xs.reshape(-1, 1))
            plt.plot(xs, ys, linewidth=2, alpha=0.8)
    plt.xlabel("BIO1 (Mean annual temperature; units per your source)")
    plt.ylabel("Longevity proxy (max series years)")
    plt.title("Longevity vs BIO1 by plant group")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(PLOTS / "scatter_longevity_bio1_by_group.png")

# 2) Pairwise scatter panels: longevity vs each BIO variable
fig, axes = plt.subplots(1, len(feature_cols), figsize=(5 * len(feature_cols), 5))
if len(feature_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, feature_cols):
    ax.scatter(X[col], y, alpha=0.8)
    ax.set_xlabel(col)
    ax.set_ylabel("Longevity proxy")
    ax.grid(True, alpha=0.3)
fig.suptitle("Longevity vs climate variables", y=1.02, fontsize=12)
savefig(PLOTS / "scatter_panels_longevity_vs_climate.png")

# 3) Boxplot: longevity by plant group
plt.figure()
labels = []
vals = []
for g in sorted(dat["plant_group"].dropna().unique()):
    idx = dat.loc[mask, "plant_group"] == g
    labels.append(g)
    vals.append(y[idx])
plt.boxplot(vals, labels=labels, showfliers=True)
plt.ylabel("Longevity proxy")
plt.title("Longevity distributions by plant group")
savefig(PLOTS / "boxplot_longevity_by_group.png")

# 4) OLS diagnostics: residuals, Q-Q, leverage
def ols_diagnostics(result, tag):
    res = result
    fitted = res.fittedvalues
    resid = res.resid
    infl = res.get_influence()
    leverage = infl.hat_matrix_diag

    # residuals vs fitted
    plt.figure()
    plt.scatter(fitted, resid, alpha=0.8)
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("Fitted")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs fitted ({tag})")
    plt.grid(True, alpha=0.3)
    savefig(PLOTS / f"diagnostics_resid_fitted_{tag}.png")

    # Q-Q plot
    plt.figure()
    sm.qqplot(resid, line="45", fit=True)
    plt.title(f"Q-Q plot of residuals ({tag})")
    savefig(PLOTS / f"diagnostics_qq_{tag}.png")

    # Leverage vs residuals^2
    plt.figure()
    plt.scatter(leverage, resid**2, alpha=0.8)
    plt.xlabel("Leverage")
    plt.ylabel("Residuals^2")
    plt.title(f"Leverage vs residuals² ({tag})")
    plt.grid(True, alpha=0.3)
    savefig(PLOTS / f"diagnostics_leverage_{tag}.png")

# ols_diagnostics(ols_res, "ols_base")
ols_diagnostics(ols2_res, "ols_poly")


# 5) Partial dependence (top features) from Random Forest
top_feats = rf_importance.sort_values(ascending=False).head(2).index.tolist()
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(rf, X, features=top_feats, ax=ax)
    plt.suptitle("Partial dependence (Random Forest)")
    savefig(PLOTS / "partial_dependence_rf.png")
except Exception as e:
    warnings.warn(f"Partial dependence plot skipped: {e}")

print("All modeling and plots complete.")
