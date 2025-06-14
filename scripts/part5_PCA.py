from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer.rotator import Rotator   
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib

class VarimaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, R):
        self.R = R
    def fit(self, X, y=None): return self
    def transform(self, X):   return X @ self.R

# ---------------------- CONFIG --------------------------------------------
CSV_PATH = Path(
    r"Path\to\your\dataset.csv" 
)
OUT_DIR = Path(r"Path\to\output\directory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 16 variables (10 originals + 6 noves) -------------------------
LIKERT_COLS = [
    # ús / coneixement
    "freq_ús", "coneixement",
    # percepció d’impacte
    "comprensio", "retenció", "rendiment", "profunditat",
    # actitud / dependència
    "confiança", "correlacio", "dependencia",
    # ítems Sí–Potser–No
    "millora_experiencia", "influencia_estrategia",
    "preocupacio_critic", "recomanaria_IA",
    # situacionals
    "impacte_semestre", "tria_examen",
    # perfil
    "anys_centre"
]

VAR_TARGET = 0.80       # objectiu ≥ 80 % var. acumulada
MIN_RESP   = 10         # mínim respostes vàlides abans d’imputar
MAX_PCS    = 4          # màxim PC que volem interpretar

# ---------------------- 1. CARREGA I NETEJA -------------------------------
warnings.filterwarnings("ignore", "Downcasting")  

raw = pd.read_csv(CSV_PATH, encoding="utf-8")

# 1.1 només estudiants
df = raw[raw["Quin és el teu rol a la teva institució educativa?"]
           .str.lower().str.contains("estudiant", na=False)].copy()

# 1.2 renombrat curt (s’inclouen les 16 preguntes)
map_curts = {  
    "En una escala de l’1 al 5, amb quina freqüència utilitzes eines d’IA (per exemple, ChatGPT) per a tasques acadèmiques?": "freq_ús",
    "En una escala de l’1 al 5, com valoraries el teu coneixement i comprensió de les eines d’IA (per exemple, ChatGPT) per a finalitats acadèmiques?": "coneixement",
    "En quina mesura creus que les eines d’IA t’ajuden a comprendre temes acadèmics complexos?": "comprensio",
    "Després d’estudiar amb assistència d’IA, quant de temps retens normalment el material après en comparació amb mètodes d’estudi tradicionals?": "retenció",
    "Creus que les eines d’IA (per exemple, ChatGPT) milloren la teva experiència general d’aprenentatge?": "millora_experiencia",
    "En una escala de l’1 al 5, en quina mesura creus que les eines d’IA milloren el teu rendiment acadèmic?": "rendiment",
    "Com de segur et sents explicant el material que has après a classe sense cap ajuda externa (incloent-hi eines d’IA)?": "confiança",
    "Creus que la teva comprensió de temes complexos és més profunda quan estudies amb eines d’IA en comparació amb mètodes d’estudi tradicionals (sense IA)?": "profunditat",
    "Al teu entendre, com de forta és la correlació entre la freqüència d’ús d’eines d’IA i el teu rendiment acadèmic (per exemple, notes d’exàmens, puntuacions de tasques)?": "correlacio",
    "En quina mesura estàs d’acord amb la següent afirmació: 'Quan faig servir eines d’IA amb regularitat, tendeixo a dependre més d’elles per resoldre problemes i aprendre, en lloc de fer-ho amb les meves pròpies habilitats cognitives.'": "dependencia",
    "L’ús d’eines d’IA ha influït en la teva manera d’abordar problemes acadèmics complexos o tasques?": "influencia_estrategia",
    "En quina mesura estàs d’acord amb la següent afirmació: 'Confiar massa en les eines d’IA (com ChatGPT) podria danyar la creativitat i les habilitats de pensament crític dels estudiants.'": "preocupacio_critic",
    "Quina probabilitat hi ha que recomanis l’ús d’eines d’IA per a finalitats acadèmiques als teus companys?": "recomanaria_IA",
    "Si no poguessis utilitzar eines d’IA durant un semestre, com creus que afectaria el teu rendiment acadèmic?": "impacte_semestre",
    "Si en un examen final et deixen escollir entre portar els teus apunts de l'assignatura o fer ús del ChatGPT, què triaries?": "tria_examen",
    "Quants anys portes a la teva institució educativa?": "anys_centre"
}
df.rename(columns=map_curts, inplace=True)

# 1.3 mapatge string→num (sí/ no / etc.)
yesno_map = {"Sí": 5, "Potser": 3, "No": 1}
for col in ["millora_experiencia", "influencia_estrategia",
            "preocupacio_critic", "recomanaria_IA"]:
    df[col] = df[col].replace(yesno_map)

df["tria_examen"] = df["tria_examen"].replace({
    "ChatGPT": 2, "No n'estic segur/a": 1, "Apunts": 0
})
df["anys_centre"] = df["anys_centre"].map({
    "Menys d’un any": 0, "1-2 anys": 1, "3-4 anys": 2, "5+ anys": 3
})

# 1.4 numèric
df[LIKERT_COLS] = df[LIKERT_COLS].apply(pd.to_numeric, errors="coerce")

# 1.5 filtre i imputació
mask = df[LIKERT_COLS].notna().sum(axis=1) >= MIN_RESP
pca_df = df.loc[mask, LIKERT_COLS].fillna(df[LIKERT_COLS].mean())
print(f"► Respostes mantingudes: {pca_df.shape[0]} (de {df.shape[0]})")

# ---------------------- 2. PCA --------------------------------------------
X = StandardScaler().fit_transform(pca_df)
pca_full = PCA().fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
k_opt   = np.argmax(cum_var >= VAR_TARGET) + 1
n_components = min(k_opt, MAX_PCS)
print(f"► Components seleccionats: {n_components} (expl. {cum_var[n_components-1]:.2%})")

pca = PCA(n_components=n_components).fit(X)
scores = pca.transform(X)

# rotació varimax
rot = Rotator()
loadings = rot.fit_transform(pca.components_.T)   # 16 × n_components

pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components)),
    ("varimax", VarimaxTransformer(rot.rotation_))
])

pca_pipeline.fit(pca_df) 
joblib.dump(pca_pipeline, OUT_DIR / "pipeline_pca_varimax.joblib")
import json

# Desa l’ordre de columnes utilitzat pel pipeline
with open(OUT_DIR / "features_pca_16items.json", "w", encoding="utf-8") as f:
    json.dump(LIKERT_COLS, f, ensure_ascii=False, indent=2)
print(" L’ordre de columnes s’ha desat a: features_pca_16items.json")
print(" Pipeline PCA+Varimax desat a:", OUT_DIR / "pipeline_pca_varimax.joblib")

# ---------------------- 3. FIGURES ----------------------------------------
# 3.1 scree acumulat
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(range(1, len(cum_var)+1), cum_var*100, marker="o")
ax.axhline(VAR_TARGET*100, ls="--", color="grey")
ax.set(title="Scree plot – variància acumulada",
       xlabel="Núm. components", ylabel="% variància explicada")
fig.tight_layout(); fig.savefig(OUT_DIR/"scree_plot.png", dpi=300)
plt.close(fig)

# 3.1-bis scree individual
expl = pca_full.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(6,3.5))
ax.plot(range(1, len(expl)+1), expl, marker="o", color="#f5a623", lw=2)
ax.set(title="Scree plot - variància explicada pels CP",
       xlabel="Component principal", ylabel="% variància explicada")
ax.set_ylim(0, max(expl)*1.15); ax.grid(axis="y", ls=":")
for x in range(1, len(expl)+1):
    ax.axvline(x, ymin=0, ymax=0.05, ls=":", color="grey", lw=0.6)
fig.tight_layout(); fig.savefig(OUT_DIR/"scree_plot_individual.png", dpi=300)
plt.close(fig)

# 3.2 biplot rotat (PC1-PC2)
if n_components >= 2:
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(scores[:,0], scores[:,1], alpha=.6)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgrey", lw=1)
    for i, var in enumerate(LIKERT_COLS):
        ax.arrow(0,0, loadings[i,0], loadings[i,1],
                 color="red", lw=1.5, head_width=0.04)
        ax.text(loadings[i,0]*1.1, loadings[i,1]*1.1, var, fontsize=8)
    ax.set(xlabel="PC1", ylabel="PC2",
           title="Biplot rotat (PC1 vs PC2)",
           xlim=(-1.1,1.1), ylim=(-1.1,1.1)); ax.grid(ls=":")
    fig.tight_layout(); fig.savefig(OUT_DIR/"biplot_pc1_pc2.png", dpi=300)
    plt.close(fig)

print(f"  Figures guardades a {OUT_DIR}")

# ---------------------- 4. EXPORTACIONS -----------------------------------
colnames = [f"RC{i+1}" for i in range(n_components)]
loadings_df = pd.DataFrame(loadings, columns=colnames, index=LIKERT_COLS)
loadings_df.to_csv(OUT_DIR/"pca_loadings_rotated.csv", float_format="%.3f")

scores_rot = pd.DataFrame(scores @ rot.rotation_,
                          columns=colnames, index=pca_df.index)
scores_rot.to_csv(OUT_DIR/"pca_rotated_scores.csv", float_format="%.4f")

print("\n=== Càrregues rotades (Varimax) ===")
print(loadings_df.round(3))
print("\n  Fitxers .csv exportats.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


print("\n=== Capacitat predictiva dels components vs ítems originals ===")
# ───────── Afegeix DINS el bloc diagnòstic, abans de definir y ─────────
CLUST_PATH = Path(r"Path\to\clustering_results.csv")

# 1) carreguem només la columna 'cluster'
clusters = pd.read_csv(CLUST_PATH, usecols=["cluster"])

# 2) filtrem RAW amb el mateix criteri d'estudiant
mask_students = raw["Quin és el teu rol a la teva institució educativa?"]\
                  .str.lower().str.contains("estudiant", na=False)
raw = raw.loc[mask_students].copy()             # ara raw té 166 files

# 3) enganxem l'etiqueta, coordinant l'índex
raw["cluster"] = clusters["cluster"].values

# 4) i ja podem definir y coherentment
y = raw.loc[pca_df.index, "cluster"]


# 5.1 Paràmetres comuns
rf = RandomForestClassifier(n_estimators=100,
                            class_weight="balanced",
                            random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = raw.loc[pca_df.index, "cluster"]     # mateixa etiqueta que al fitxer clustering_results.csv

def f1_macro(X, y, name):
    f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro")
    print(f"{name:<12}: F1 = {f1.mean():.3f} ± {f1.std():.3f}")
    return f1.mean(), f1.std()

# 5.2 F1 amb els 16 ítems bruts (ja imputats)
f1_items, sd_items = f1_macro(pca_df[LIKERT_COLS], y, "16 ítems")

# 5.3 F1 amb els RC rotats
f1_rc, sd_rc = f1_macro(scores_rot, y, "4 RC")

pd.DataFrame({
    "Model": ["RF_16_items", "RF_4_RC"],
    "F1_macro": [round(f1_items, 3), round(f1_rc, 3)],
    "SD": [round(sd_items, 3), round(sd_rc, 3)]
}).to_csv(OUT_DIR / "pca_f1_summary.csv", index=False)
print("\nResum F1 exportat a:", OUT_DIR / "pca_f1_summary.csv") 
