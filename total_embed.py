import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

# --- SETUP PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumiamo che il file sia nella root del progetto
PROJECT_ROOT = CURRENT_DIR
EMBEDDINGS_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "embeddings")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_speaker_id(filename):
    """
    Estrae lo speaker ID dal nome file.
    Esempio: 'p225_001' -> 'p225'
    Se il formato è diverso, adatta la logica split.
    """
    return filename.split('_')[0]

def load_and_normalize(path):
    """Carica un embedding e applica la normalizzazione L2."""
    try:
        emb = np.load(path)
        emb = emb.flatten() # Forza a 1D
        norm = np.linalg.norm(emb)
        if norm > 0:
            return emb / norm
        return emb
    except Exception as e:
        print(f"❌ Errore nel caricamento di {path}: {e}")
        return None

def create_database():
    """Scansiona le cartelle e crea un DataFrame con tutti i dati."""
    data_list = []
    # I nomi delle sottocartelle definiti in super_wrapper.py
    models = ["cosyvoice", "gpt_sovits", "openvoice", "styletts2"]
    
    print("📂 Caricamento degli embedding in corso...")
    
    for model in models:
        model_dir = os.path.join(EMBEDDINGS_BASE_DIR, model)
        if not os.path.exists(model_dir):
            continue
            
        files = [f for f in os.listdir(model_dir) if f.endswith('.npy')]
        for f in files:
            # Pulizia del nome file per risalire all'originale
            clean_name = f.replace("_cosy.npy", "").replace("_gpt.npy", "") \
                          .replace("_ov.npy", "").replace("_style.npy", "")
            
            emb_path = os.path.join(model_dir, f)
            embedding = load_and_normalize(emb_path)
            
            if embedding is not None:
                data_list.append({
                    "filename": clean_name,
                    "speaker": get_speaker_id(clean_name),
                    "model": model,
                    "embedding": embedding
                })
    
    return pd.DataFrame(data_list)

def plot_tsne(df):
    """Genera e salva il grafico t-SNE."""
    if len(df) < 2:
        print("⚠️ Troppi pochi dati per generare un t-SNE.")
        return

    print("🎨 Generazione del grafico t-SNE (riduzione dimensionale)...")
    
    # Normalizzazione embedding a dimensione comune usando PCA
    # Gli embedding da modelli diversi hanno dimensioni diverse
    # PCA li riduce tutti a 64 dimensioni per confrontabilità
    embeddings_list = df['embedding'].values
    
    # Trova la massima dimensione
    max_dim = max([e.shape[0] for e in embeddings_list])
    
    # PCA per normalizzare a 64D
    target_dim = min(64, max_dim - 1)  # Min 64 o meno se embedding piccoli
    
    try:
        pca = PCA(n_components=target_dim)
        embeddings_normalized = pca.fit_transform(np.array([e.flatten() for e in embeddings_list]))
        print(f"   ✓ Embedding normalizzati a {target_dim}D via PCA")
    except Exception as e:
        print(f"⚠️  Errore PCA: {e}. Uso embedding raw.")
        embeddings_normalized = np.array([e.flatten() for e in embeddings_list])
    
    # t-SNE su embedding normalizzati
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
    vis_dims = tsne.fit_transform(embeddings_normalized)
    
    # Aggiungiamo le coordinate al DataFrame
    df['x'] = vis_dims[:, 0]
    df['y'] = vis_dims[:, 1]
    
    # Creazione del Plot
    plt.figure(figsize=(14, 10))
    
    # Visualizziamo i punti colorati per modello e con forme diverse per speaker (se non troppi)
    # Se hai troppi speaker, usa solo 'hue' per il modello
    plot = sns.scatterplot(
        data=df, 
        x='x', y='y', 
        hue='model', 
        style='model', 
        s=100, 
        palette='viridis',
        alpha=0.7
    )
    
    plt.title('Voice Identity Analysis: t-SNE Cluster Comparison', fontsize=15)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(OUTPUT_DIR, "tsne_analysis.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Grafico salvato con successo in: {plot_path}")

def main():
    # 1. Creazione Database
    df = create_database()
    
    if df.empty:
        print("❌ Database vuoto. Assicurati che gli embedding siano in data/embeddings/")
        return
        
    print(f"✅ Caricati {len(df)} embedding.")

    # 2. Salvataggio Database (per usi futuri in altri script)
    pkl_path = os.path.join(OUTPUT_DIR, "embedding_db.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"✅ Database salvato in: {pkl_path}")

    # 3. Analisi Grafica
    plot_tsne(df)

if __name__ == "__main__":
    main()