import warnings
import pandas as pd
import statsmodels.api as sm
from textwrap import shorten

# —————————————————— CONFIGURACIÓ ——————————————————
CSV_PATH = (
    r"Path\to\your\dataset.csv" 
)


warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# —————————————————— FUNCIONS AUXILIARS ——————————————————
def regressio_lineal(df, y_col, x_cols, extra=""):
    tmp = df.dropna(subset=[y_col] + x_cols)
    n = len(tmp)
    if n < 10:                                      
        print(f"  Omet {y_col}{extra}: {n} observacions")
        return
    y = tmp[y_col]
    X = sm.add_constant(tmp[x_cols])
    model = sm.OLS(y, X).fit()
    print(f"\n🔷 Lineal: {y_col} ~ {x_cols} {extra} (n={n})")
    print(f"R² aj. = {model.rsquared_adj:.3f}")
    print(model.summary().tables[1])


def regressio_logistica(df, y_col, x_cols, threshold=4, ja_binaritzat=False, extra=""):
    tmp = df.dropna(subset=[y_col] + x_cols)
    n = len(tmp)
    if n < 10:
        print(f"  Omet {y_col}{extra}: {n} observacions")
        return
    y_bin = tmp[y_col] if ja_binaritzat else (tmp[y_col] >= threshold).astype(int)
    X = sm.add_constant(tmp[x_cols])
    model = sm.Logit(y_bin, X).fit(disp=False)
    pseudo_r2 = 1 - model.llf / model.llnull
    print(f"\n Logística: {y_col} ~ {x_cols} {extra}  (n={n})")
    print(f"Pseudo-R² = {pseudo_r2:.3f}")
    print(model.summary().tables[1])


def mostra_valors_uniques(df, cols, max_len=55):
    print("\n── Mostra valors únics actuals ──")
    for c in cols:
        vals = ", ".join(map(str, df[c].dropna().unique()))
        print(f"{c:<18}: {shorten(vals, max_len)}")

# —————————————————— LECTURA I FILTRAT ——————————————————
df = pd.read_csv(CSV_PATH)
print(f"\n── Després de carregar CSV ──  {df.shape}")

df = df[df["Quin és el teu rol a la teva institució educativa?"]
        .str.lower().str.contains("estudiant", na=False)].copy()
print(f"── Després de filtre 'estudiant' ──  {df.shape}")

# —————————————————— RENOMBRAT CURT ——————————————————
col_map = {
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
    "Si en un examen final et deixen escollir entre portar els teus apunts de l'assignatura o fer ús del ChatGPT, què triaries?": "tria_examen",
    "Si no poguessis utilitzar eines d’IA durant un semestre, com creus que afectaria el teu rendiment acadèmic?": "impacte_semestre",
}
df.rename(columns=col_map, inplace=True)

# —————————————————— MAPATGES (sense inplace) ——————————————————
retencio_map = {
    "Menys temps que amb mètodes tradicionals": 1,
    "Una mica menys de temps": 2,
    "El mateix temps": 3,
    "Una mica més de temps": 4,
    "Més temps que amb mètodes tradicionals": 5,
}
impacte_map = {
    "No m'afectaria gens": 5,
    "M'afectaria una mica": 4,
    "M'afectaria moderadament": 3,
    "M'afectaria bastant": 2,
    "M'afectaria molt": 1,
}
millora_map = {"No": 1, "Potser": 3, "Sí": 5}

maps = {
    "retenció": retencio_map,
    "impacte_semestre": impacte_map,
    "millora_experiencia": millora_map,
}
for col, mp in maps.items():
    if col in df.columns:
        df[col] = df[col].replace(mp)

# —————————————————— CONVERSIÓ NUMÈRICA ——————————————————
numeric_cols = list(maps.keys()) + [
    "freq_ús", "coneixement", "comprensio", "rendiment", "confiança",
    "correlacio", "dependencia", "profunditat"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")

# —————————————————— BINARITZACIÓ TRIA EXAMEN ——————————————————
VALORS_BONS_TRIA = ["Apunts", "ChatGPT"]
df = df[df["tria_examen"].isin(VALORS_BONS_TRIA)].copy()
df["tria_chatgpt"] = (df["tria_examen"] == "ChatGPT").astype(int)

# —————————————————— DEBUG CURT ——————————————————
mostra_valors_uniques(df, numeric_cols + ["tria_examen"])

print(f"\n── Després de filtrar tria_examen ── {df.shape}")

# —————————————————— MODELS ——————————————————
regressio_lineal(df, "comprensio", ["freq_ús", "coneixement"])
regressio_lineal(df, "retenció",   ["freq_ús", "coneixement", "millora_experiencia"])
regressio_lineal(df, "confiança",  ["freq_ús", "coneixement"])
regressio_lineal(df, "rendiment",  ["freq_ús", "coneixement", "correlacio"])

regressio_logistica(df, "dependencia",
                    ["freq_ús", "coneixement", "millora_experiencia"])

regressio_logistica(df, "profunditat",
                    ["freq_ús", "coneixement", "correlacio"])

# — 4.1 ———————————————————————————————————————————
regressio_lineal(
    df,
    "rendiment",
    ["coneixement", "freq_ús", "comprensio", "dependencia"],
    extra=" (Model informe 4.1)"
)

# — 4.2 ———————————————————————————————————————————
regressio_logistica(
    df,
    "tria_chatgpt",
    ["freq_ús", "coneixement", "impacte_semestre", "confiança", "dependencia"],
    ja_binaritzat=True,
    extra=" (Model informe 4.2)"
)

# —————————————————— DISTRIBUCIONS FINALS ——————————————————
print("\nDistribució 'dependencia':")
print(df["dependencia"].value_counts(dropna=False))