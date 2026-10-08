import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# All data below is simulated and all rates are assumptions.

# ---- 1. Baseline vs improved loss model (all rates ASSUMED) ----
orders = 1500  # per week

def lost(delay_days, comm_loss, refund_loss, per_day=0.04):
    d = orders * per_day * delay_days
    c = orders * comm_loss
    r = orders * refund_loss
    return d, c, r, d + c + r

base = lost(3, 0.03, 0.05)
better = lost(1.5, 0.015, 0.02)
df_loss = pd.DataFrame([base, better], index=["Baseline", "Improved"],
    columns=["Delay", "Communication", "Damage", "Total lost"])
df_loss["Retained"] = orders - df_loss["Total lost"]
print(df_loss.round(1))

# ---- 1b. Sensitivity check: ASSUMED rates changed by 20% ----
for s in (0.8, 1.2):  # all rates lower / higher
    b = lost(3, 0.03*s, 0.05*s, per_day=0.04*s)
    i = lost(1.5, 0.015*s, 0.02*s, per_day=0.04*s)
    print(f"All rates x{s}: baseline {b[3]:.0f}, improved {i[3]:.1f}, "
          f"cut {1 - i[3]/b[3]:.1%}")

for s in (0.8, 1.2):  # only the improved-case rates change
    i = lost(1.5, 0.015*s, 0.02*s, per_day=0.04*s)
    print(f"Improved rates x{s}: improved {i[3]:.1f}, "
          f"cut {1 - i[3]/base[3]:.1%}")

# ---- 2. SIMULATED order data (not real Blue Dart data) ----
rng = np.random.default_rng(42)
n = 1500
df = pd.DataFrame({
    "distance_km": rng.gamma(3, 150, n),
    "weight_kg": rng.gamma(2, 2, n),
    "fragile": rng.integers(0, 2, n),
    "delay_days": rng.poisson(3, n).astype(float),  # baseline average: 3 days
})
df.loc[rng.choice(n, 40, replace=False), "weight_kg"] = np.nan
df.loc[rng.choice(n, 10, replace=False), "distance_km"] = -5
df = pd.concat([df, df.iloc[:20]])

# ---- 3. Cleaning ----
df = df.drop_duplicates()
df = df[df["distance_km"] > 0].copy()
df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())

# ---- 4. Clustering: group orders by profile ----
X = StandardScaler().fit_transform(df[["distance_km", "weight_kg", "delay_days"]])
df["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
print(df.groupby("cluster")[["distance_km", "weight_kg", "delay_days"]].mean().round(1))

# ---- 5. Classification: predict late orders (3+ days) ----
df["late"] = (df["delay_days"] >= 3).astype(int)
feats = ["distance_km", "weight_kg", "fragile"]
Xtr, Xte, ytr, yte = train_test_split(df[feats], df["late"],
                                      test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, pred), 3))
# Logistics-
