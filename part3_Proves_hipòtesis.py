import pandas as pd
import numpy as np
import os
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURACIÓ ===
CSV_PATH = r"Path\to\your\dataset.csv" 
FIGURES_DIR = r"Path\to\your\figures_directory"
os.makedirs(FIGURES_DIR, exist_ok=True)

# === FUNCIONS AUXILIARS ===
def definir_grups_us(df, col_frequencia, llindar_intensiu):
    df['grup_ús'] = np.where(df[col_frequencia] >= llindar_intensiu, 'Intensiu', 'No intensiu')

def welch_ttest_independent(df, grup_col, valor_col, valor_grup1, valor_grup2):
    grup1 = df[df[grup_col] == valor_grup1][valor_col].dropna()
    grup2 = df[df[grup_col] == valor_grup2][valor_col].dropna()
    if len(grup1) < 2 or len(grup2) < 2:
        print(f"No hi ha suficients mostres per al t-test en {valor_col}.")
        return None
    t_stat, p_val = stats.ttest_ind(grup1, grup2, equal_var=False)
    print(f"Comparació {valor_col} entre {valor_grup1} vs {valor_grup2}: t = {t_stat:.3f}, p-value = {p_val:.3f}")
    return t_stat, p_val

def plot_boxplot(df, grup_col, valor_col, save_path=None):
    plt.figure(figsize=(6,4))
    sns.boxplot(x=grup_col, y=valor_col, data=df, palette="pastel", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"})
    plt.title(f"Distribució de '{valor_col}' per grups de {grup_col}")
    plt.xlabel(grup_col)
    plt.ylabel(valor_col)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# === CÀRREGA I PREPARACIÓ DE DADES ===
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"Error en carregar el dataset: {e}")
    df = pd.DataFrame()

df = df[df["Quin és el teu rol a la teva institució educativa?"].str.lower().str.contains("estudiant", na=False)].copy()

columnes = {
    "En una escala de l’1 al 5, amb quina freqüència utilitzes eines d’IA (per exemple, ChatGPT) per a tasques acadèmiques?": "freq_ús",
    "En quina mesura creus que les eines d’IA t’ajuden a comprendre temes acadèmics complexos?": "comprensio",
    "Com de segur et sents explicant el material que has après a classe sense cap ajuda externa (incloent-hi eines d’IA)?": "confiança",
    "Després d’estudiar amb assistència d’IA, quant de temps retens normalment el material après en comparació amb mètodes d’estudi tradicionals?": "retenció",
    "Creus que la teva comprensió de temes complexos és més profunda quan estudies amb eines d’IA en comparació amb mètodes d’estudi tradicionals (sense IA)?": "profunditat"
}

df.rename(columns=columnes, inplace=True)

for col in columnes.values():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

definir_grups_us(df, col_frequencia="freq_ús", llindar_intensiu=4)

variables_interes = ["comprensio", "confiança", "retenció", "profunditat"]

for var in variables_interes:
    if df.empty or var not in df.columns:
        continue
    welch_ttest_independent(df, "grup_ús", var, valor_grup1="Intensiu", valor_grup2="No intensiu")
    save_file = os.path.join(FIGURES_DIR, f"boxplot_{var}.png")
    plot_boxplot(df, "grup_ús", var, save_path=save_file)
