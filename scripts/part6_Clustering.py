from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ----------------------- CONFIG --------------------------------------
CSV_RAW   = Path(r"Path\to\your\raw_data.csv")  
PCA_SCORES = Path(r"Path\to\your\pca_scores.csv") 
OUT_DIR    = Path(r"Path\to\your\output_directory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RC_COLS = [f"RC{i}" for i in range(1, 5)]        # 4 components finals
N_CLUSTERS = 4                                  

# ----------------------- 1. DADES ------------------------------------
raw = pd.read_csv(CSV_RAW)
df_students = raw[raw["Quin és el teu rol a la teva institució educativa?"]
                  .str.lower().str.contains("estudiant", na=False)].copy()

scores = pd.read_csv(PCA_SCORES, index_col=0)  
df = df_students.join(scores[RC_COLS], how="inner")

X = StandardScaler().fit_transform(df[RC_COLS])
print(f"  Matriu per agrupar: {X.shape[0]} mostres × {X.shape[1]} RC")

# ----------------------- 2. K-MEANS ----------------------------------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)

sil = silhouette_score(X, labels)
print(f"\nSilhouette global = {sil:.3f}  (k = {N_CLUSTERS})")

df["cluster"] = labels
print("\nMida de cada clúster:")
print(df["cluster"].value_counts().rename("n_membres"))

print("\nMitjana de les RC per clúster (z-scores):")
print(df.groupby("cluster")[RC_COLS].mean().round(2))

# ----------------------- 3. ELBOW + SILHOUETTE (opc.) ---------------
def elbow_and_silhouette(X, k_max=10):
    inert, silh = [], []
    ks = range(2, k_max + 1)
    for k in ks:
        lbl = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X)
        inert.append(KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X).inertia_)
        silh.append(silhouette_score(X, lbl))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ks, inert, 'bo-')
    ax1.set_xlabel("k"); ax1.set_ylabel("Inèrcia (SSE)", color='b')
    ax2 = ax1.twinx()
    ax2.plot(ks, silh, 'ro-'); ax2.set_ylabel("Silhouette", color='r')
    ax1.axvline(N_CLUSTERS, ls='--', c='grey')
    fig.tight_layout(); fig.savefig(OUT_DIR / "elbow_silhouette.png", dpi=300)
    plt.close(fig)

elbow_and_silhouette(X)

# ----------------------- 4. GUARDA RESULTATS -------------------------
plt.figure(figsize=(7, 5))
for c in range(N_CLUSTERS):
    plt.scatter(
        df.loc[df["cluster"] == c, "RC2"],
        df.loc[df["cluster"] == c, "RC3"],
        label=f"Clúster {c}",
        alpha=0.7
    )
plt.xlabel("RC2")
plt.ylabel("RC3")
plt.title("Visualització dels clústers (RC2 vs RC3)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "clusters_scatter_R2_R3.png", dpi=300)
plt.show()


from mpl_toolkits.mplot3d import Axes3D  

# Plot 3D dels clústers (RC1, RC2, RC3)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

for c in range(N_CLUSTERS):
    ax.scatter(
        df.loc[df["cluster"] == c, "RC1"],
        df.loc[df["cluster"] == c, "RC2"],
        df.loc[df["cluster"] == c, "RC3"],
        label=f"Clúster {c}",
        alpha=0.7,
        color=colors[c % len(colors)]
    )
    # Centroide de cada clúster
    ax.scatter(
        kmeans.cluster_centers_[c, 0],  # RC1
        kmeans.cluster_centers_[c, 1],  # RC2
        kmeans.cluster_centers_[c, 2],  # RC3
        c=colors[c % len(colors)],
        marker='X',
        s=300,
        edgecolor='black',
        label=f"Centroide {c}"
    )

ax.set_xlabel("RC1")
ax.set_ylabel("RC2")
ax.set_zlabel("RC3")
ax.set_title("Visualització 3D dels clústers i centroides (RC1, RC2, RC3)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "clusters_scatter_3d_centroides.png", dpi=300)
plt.show()

# ----------------------- 5. GUARDA PIPELINE --------------------------
from sklearn.pipeline import Pipeline
import joblib

pipeline = Pipeline([
    ("scaler", StandardScaler()),  
    ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto'))
])
pipeline.fit(df[RC_COLS])  

joblib.dump(pipeline, OUT_DIR / "pipeline_kmeans_rc_scaled.joblib")
print(" Pipeline desat a:", OUT_DIR / "pipeline_kmeans_rc_scaled.joblib")