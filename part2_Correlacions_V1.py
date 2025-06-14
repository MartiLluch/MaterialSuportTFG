import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import pearsonr, spearmanr
import os
from itertools import combinations

# === CONFIGURACIÓ ===
CSV_PATH = r"Path\to\your\file.csv" 
LLINDAR_CORRELACIO = 0.3
GRUP_OBJECTIU = "estudiant" 
EXPORTAR_CSV = True
OUTPUT_DIR = r"Path\to\output\directory" 

# === CARREGA I FILTRAT ===
df = pd.read_csv(CSV_PATH)
df_filtrat = df[df["Quin és el teu rol a la teva institució educativa?"]
                .str.lower().str.contains(GRUP_OBJECTIU, na=False)]

# === CODIFICACIÓ DE COLUMNES NO-LIKERT ===
df_codificat = df_filtrat.copy()

# 1. Anys a la institució
map_anys = {
    "Menys d’un any": 0,
    "1-2 anys": 1,
    "3-4 anys": 2,
    "5+ anys": 3
}
df_codificat["anys_institucio"] = df_codificat["Quants anys portes a la teva institució educativa?"].map(map_anys)

# 2. Contextos d'ús
df_codificat["contextos_IA"] = df_codificat[
    "En quins dels següents contextos utilitzes eines d’IA? (Selecciona totes les opcions aplicables)"]
contextos = df_codificat["contextos_IA"].dropna().str.split(";").explode().str.strip().unique().tolist()
for ctx in contextos:
    df_codificat[f"context_{ctx}"] = df_codificat["contextos_IA"].fillna("").apply(lambda x: int(ctx in x))

# 3. Preferència examen
map_preferencia = {
    "Apunts": 0,
    "No n'estic segur/a": 1,
    "ChatGPT": 2
}
col_examen = "Si en un examen final et deixen escollir entre portar els teus apunts de l'assignatura o fer ús del ChatGPT, què triaries?"
df_codificat["preferencia_chatgpt"] = df_codificat[col_examen].map(map_preferencia)

# === COMBINACIÓ AMB COLUMES LIKERT ===
df_numeric = df_codificat.copy()
for col in df_numeric.columns:
    try:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    except:
        continue

likert_cols_detectades = [
    col for col in df_numeric.columns
    if df_numeric[col].dropna().between(1, 5).all() and df_numeric[col].nunique() <= 5
]

# Columnes afegides (sense usa_* ni area_*)
columnes_afegides = ["anys_institucio", "preferencia_chatgpt"] + \
                    [col for col in df_codificat.columns if col.startswith("context_")]

df_likert = df_numeric[likert_cols_detectades + columnes_afegides].dropna(axis=1, how='all')

# === VALIDACIÓ ===
if df_likert.shape[1] == 0:
    raise ValueError("No s’han trobat columnes vàlides per a l’anàlisi.")
if df_likert.shape[0] < 10:
    raise ValueError("No hi ha prou respostes per fer anàlisi fiable.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FUNCIONS ===
def ordena_variables(corr_matrix):
    linkage_matrix = linkage(corr_matrix.fillna(0), method='ward')
    ordered_idx = leaves_list(linkage_matrix)
    return corr_matrix.iloc[ordered_idx, ordered_idx]

def guarda_heatmap(corr_matrix, title, nom_fitxer):
    ordered = ordena_variables(corr_matrix)
    plt.figure(figsize=(14, 12))
    sns.heatmap(ordered, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Coeficient de correlació'})
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    ruta = os.path.join(OUTPUT_DIR, nom_fitxer)
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    print(f" Figura desada: {ruta}")
    return ordered

def calcula_correlacions_significatives(df, metode):
    resultats = []
    df = df.select_dtypes(include=["number"]).copy()
    for var1, var2 in combinations(df.columns, 2):
        df_common = df[[var1, var2]].dropna()
        if len(df_common) > 10 and df_common[var1].ndim == 1 and df_common[var2].ndim == 1:
            x, y = df_common[var1], df_common[var2]
            try:
                if metode == 'pearson':
                    r, p = pearsonr(x, y)
                else:
                    r, p = spearmanr(x, y)
                if abs(r) >= LLINDAR_CORRELACIO:
                    resultats.append({
                        "Variable 1": var1,
                        "Variable 2": var2,
                        "Coeficient": r,
                        "p-value": p
                    })
            except Exception as e:
                print(f" Error amb {var1} i {var2}: {e}")
                continue
    return pd.DataFrame(resultats).sort_values(by="Coeficient", ascending=False)

# === ANÀLISI DE CORRELACIONS AMB SIGNIFICACIÓ ===
corr_pearson = df_likert.corr(method='pearson')
corr_spearman = df_likert.corr(method='spearman')

pearson_significatives = calcula_correlacions_significatives(df_likert, 'pearson')
spearman_significatives = calcula_correlacions_significatives(df_likert, 'spearman')

guarda_heatmap(corr_pearson, f"Mapa de calor (Pearson) - {GRUP_OBJECTIU.capitalize()}",
               f"heatmap_pearson_{GRUP_OBJECTIU}.png")
guarda_heatmap(corr_spearman, f"Mapa de calor (Spearman) - {GRUP_OBJECTIU.capitalize()}",
               f"heatmap_spearman_{GRUP_OBJECTIU}.png")

# === RESULTATS TERMINAL ===
print("\n Correlacions significatives (Pearson):")
print(pearson_significatives)

print("\n Correlacions significatives (Spearman):")
print(spearman_significatives)

# === EXPORTACIÓ ===
if EXPORTAR_CSV:
    df_likert.to_csv(os.path.join(OUTPUT_DIR, f"respostes_{GRUP_OBJECTIU}.csv"), index=False)
    pearson_significatives.to_csv(os.path.join(OUTPUT_DIR, f"correlacions_pearson_{GRUP_OBJECTIU}_amb_pval.csv"), index=False)
    spearman_significatives.to_csv(os.path.join(OUTPUT_DIR, f"correlacions_spearman_{GRUP_OBJECTIU}_amb_pval.csv"), index=False)
    print(f"\n Fitxers .csv desats a /{OUTPUT_DIR}/")
