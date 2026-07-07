import os
import re
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# I dati possono stare su un disco diverso (es. /media/SSD_new/...): basta
# esportare VOICE_DATA_DIR prima di lanciare, senza toccare il codice.
DATA_DIR = os.environ.get("VOICE_DATA_DIR", os.path.join(CURRENT_DIR, "data"))
DB_PATH = os.path.join(DATA_DIR, "embedding_db.pkl")
OUTPUT_CSV = os.path.join(DATA_DIR, "statistical_results.csv")
PER_SYSTEM_CSV = os.path.join(DATA_DIR, "statistical_results_per_system.csv")

# Formato dei nomi file del dataset ITASpoof:
#   detection_{generatore}_speaker{speakerID}_sentence{sentenceID}.wav
# 'natural' = audio reale (label 0); ogni altro generatore = sintetico (system 1-7).
# 'generator' e 'speaker' sono non-greedy per reggere eventuali underscore interni.
FILENAME_RE = re.compile(
    r"detection_(?P<generator>.+?)_speaker(?P<speaker>.+?)_sentence(?P<sentence>[^_.]+)",
    re.IGNORECASE,
)


def load_data():
    """Carica il database degli embedding."""
    if not os.path.exists(DB_PATH):
        print(f"Database non trovato: {DB_PATH}")
        print("Esegui prima lo step di estrazione degli embedding.")
        return None
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def parse_metadata(df):
    """Estrae generatore, speaker, sentence e tipo (Real/Fake) dal filename."""
    def parse_one(filename):
        base = os.path.splitext(os.path.basename(str(filename)))[0]
        m = FILENAME_RE.match(base)
        if not m:
            return pd.Series({
                "generator": None, "speaker_id": None,
                "sentence_id": None, "voice_type": "Unknown",
            })
        gen = m.group("generator").lower()
        return pd.Series({
            "generator": gen,
            "speaker_id": m.group("speaker"),
            "sentence_id": m.group("sentence"),
            "voice_type": "Real" if gen == "natural" else "Fake",
        })

    meta = df["filename"].apply(parse_one)
    df = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)

    n_unknown = int((df["voice_type"] == "Unknown").sum())
    if n_unknown:
        print(f"⚠️  {n_unknown} file non riconosciuti dal parser dei nomi (ignorati).")
        # Mostra un paio di esempi per capire subito se il formato è diverso.
        sample = df[df["voice_type"] == "Unknown"]["filename"].head(3).tolist()
        if sample:
            print(f"    Esempi: {sample}")

    return df[df["voice_type"] != "Unknown"].copy()


def calculate_dispersion(embeddings):
    """Dispersione (varianza media attorno al centroide) di un insieme di embedding."""
    if len(embeddings) < 2:
        return 0.0
    matrix = np.stack(embeddings)
    centroid = np.mean(matrix, axis=0)
    distances = [euclidean(vec, centroid) ** 2 for vec in matrix]
    return float(np.mean(distances))


def cosine_matched(real_df, fake_df):
    """
    Cosine similarity tra ogni fake e il SUO reale corrispondente,
    accoppiati per (speaker_id, sentence_id): stesso enunciato, reale vs clone.
    """
    real_lookup = {
        (row["speaker_id"], row["sentence_id"]): row["embedding"]
        for _, row in real_df.iterrows()
    }
    sims = []
    for _, row in fake_df.iterrows():
        ref = real_lookup.get((row["speaker_id"], row["sentence_id"]))
        if ref is not None:
            sims.append(1 - cosine(ref, row["embedding"]))
    return sims


def run_statistical_analysis(df):
    print("Analisi statistica dello spazio latente")
    print("=" * 60)

    overall_rows = []
    per_system_rows = []

    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        real_df = mdf[mdf["voice_type"] == "Real"]
        fake_df = mdf[mdf["voice_type"] == "Fake"]

        if real_df.empty or fake_df.empty:
            print(f"Modello {model}: mancano reali o sintetici, salto.")
            continue

        real_disp = calculate_dispersion(real_df["embedding"].values)
        fake_disp = calculate_dispersion(fake_df["embedding"].values)
        ratio = fake_disp / real_disp if real_disp > 0 else 0.0
        sims = cosine_matched(real_df, fake_df)
        avg_sim = float(np.mean(sims)) if sims else 0.0

        overall_rows.append({
            "Model": model,
            "Real_Var": round(real_disp, 4),
            "Fake_Var": round(fake_disp, 4),
            "Entropy_Ratio(F/R)": round(ratio, 4),
            "Avg_Sim": round(avg_sim, 4),
            "Pairs": len(sims),
        })

        # Dettaglio per singolo sistema di sintesi: serve a cercare il pattern
        # che distingue una sorgente di cloning dall'altra guardando l'embedding.
        for gen in sorted(fake_df["generator"].unique()):
            gdf = fake_df[fake_df["generator"] == gen]
            g_disp = calculate_dispersion(gdf["embedding"].values)
            g_sims = cosine_matched(real_df, gdf)
            per_system_rows.append({
                "Model": model,
                "System": gen,
                "Fake_Var": round(g_disp, 4),
                "Entropy_Ratio(F/R)": round(g_disp / real_disp if real_disp > 0 else 0.0, 4),
                "Avg_Sim": round(float(np.mean(g_sims)) if g_sims else 0.0, 4),
                "N": len(gdf),
            })

    if not overall_rows:
        print("Nessun risultato. Controlla i nomi dei file / il parser.")
        return

    overall = pd.DataFrame(overall_rows)
    print("\nRIEPILOGO PER ENCODER (reale vs tutti i sintetici):\n")
    print(overall.to_string(index=False))

    per_sys = None
    if per_system_rows:
        per_sys = pd.DataFrame(per_system_rows)
        print("\nDETTAGLIO PER SISTEMA DI SINTESI:\n")
        print(per_sys.to_string(index=False))

    print("\n" + "-" * 60)
    print("LETTURA:")
    print("  Entropy_Ratio < 1  -> i sintetici occupano meno 'volume' nello spazio")
    print("                        latente dei reali (ipotesi 'reduced entropy').")
    print("  Avg_Sim vicino a 1 -> l'encoder vede reale e clone quasi identici.")
    print("-" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)
    overall.to_csv(OUTPUT_CSV, index=False)
    print(f"Risultati salvati in: {OUTPUT_CSV}")
    if per_sys is not None:
        per_sys.to_csv(PER_SYSTEM_CSV, index=False)
        print(f"Dettaglio per sistema salvato in: {PER_SYSTEM_CSV}")


def main():
    df = load_data()
    if df is not None:
        df = parse_metadata(df)
        run_statistical_analysis(df)


if __name__ == "__main__":
    main()