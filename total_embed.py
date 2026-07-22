import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "src"))

from metadata_utils import attach_metadata

# VOICE_DATA_DIR (impostata in run_analysis.sh) decide dove leggere il database
# e dove salvare i grafici. Default: ./data accanto agli altri output.
DATA_DIR = os.environ.get("VOICE_DATA_DIR", "data")


def load_embeddings(db_path):
    if not os.path.exists(db_path):
        return None
    with open(db_path, "rb") as f:
        return pickle.load(f)


def plot_voice_comparison(df, output_path):
    models = df["model"].unique()
    n = len(models)
    # Griglia adattiva (2x2 con 4 modelli, ma regge anche numeri diversi).
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 6 * rows), squeeze=False)
    axes = axes.flatten()

    print("🎨 Generazione grafici PCA...")

    # Colori per categoria: reale ben distinto, sintetici in palette separata.
    color_map = {"Real": "tab:green", "Fake": "tab:red", "Unknown": "tab:gray"}

    for i, model_name in enumerate(models):
        ax = axes[i]
        model_df = df[df["model"] == model_name].copy()

        embeddings = np.array([e.flatten() for e in model_df["embedding"].values])
        # Micro-rumore per evitare colonne a varianza zero nello StandardScaler.
        embeddings = embeddings + np.random.normal(0, 1e-8, embeddings.shape)
        embeddings_scaled = StandardScaler().fit_transform(embeddings)

        coords = PCA(n_components=2).fit_transform(embeddings_scaled)

        # Un colore per voice_type; niente annotazioni per-punto (illeggibili su
        # dataset grandi). La legenda riassume le categorie.
        for vtype in ["Real", "Fake", "Unknown"]:
            mask = (model_df["voice_type"] == vtype).values
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    label=vtype, s=25, alpha=0.6,
                    color=color_map[vtype], edgecolors="none",
                )

        ax.set_title(f"Modello: {model_name.upper()}", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("Componente Principale 1")
        ax.set_ylabel("Componente Principale 2")
        ax.legend(loc="best", fontsize=9)

    # Nasconde eventuali assi in eccesso se i modelli non riempiono la griglia.
    for k in range(len(models), len(axes)):
        axes[k].set_visible(False)

    plt.suptitle("Analisi Identità Vocale - Reale vs Sintetico", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"✅ Grafico salvato con successo in: {output_path}")
    plt.close()


def main():
    db_path = os.path.join(DATA_DIR, "embedding_db.pkl")
    output_path = os.path.join(DATA_DIR, "voice_comparison_pca.png")

    data = load_embeddings(db_path)
    if data is None or len(data) == 0:
        print(f"❌ Errore: Database {db_path} non trovato o vuoto.")
        return

    df = pd.DataFrame(data)
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(str(x)))

    # Aggiunge generator + voice_type per colorare i punti.
    parsed = df["filename"].apply(parse_name)
    df["generator"] = parsed.apply(lambda t: t[0])
    df["voice_type"] = parsed.apply(lambda t: t[1])

    n_unknown = int((df["voice_type"] == "Unknown").sum())
    if n_unknown:
        print(f"⚠️  {n_unknown} file non riconosciuti dal parser dei nomi "
              f"(verranno mostrati in grigio come 'Unknown').")

    try:
        plot_voice_comparison(df, output_path)
    except Exception as e:
        print(f"❌ Errore critico durante la generazione del grafico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()