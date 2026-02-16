import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_embeddings(db_path):
    if not os.path.exists(db_path):
        return None
    with open(db_path, 'rb') as f:
        return pickle.load(f)

def plot_voice_comparison(df, output_path="data/voice_comparison_pca.png"):
    # Creiamo una figura con 4 grafici (uno per ogni modello)
    models = df['model'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    print("🎨 Generazione grafici PCA...")

    for i, model_name in enumerate(models):
        ax = axes[i]
        # Filtriamo i dati per il modello corrente
        model_df = df[df['model'] == model_name].copy()
        
        # Estraiamo gli embedding e assicuriamoci che siano piatti
        embeddings = np.array([e.flatten() for e in model_df['embedding'].values])
        
        # 1. Normalizzazione (StandardScaler)
        # Aggiungiamo un piccolissimo rumore (epsilon) per evitare divisioni per zero 
        # se i file sono troppo simili
        embeddings = embeddings + np.random.normal(0, 1e-8, embeddings.shape)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # 2. PCA (2 componenti per il grafico 2D)
        # Usiamo PCA invece di t-SNE perché è stabile con pochi file
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings_scaled)
        
        # 3. Plot
        filenames = model_df['filename'].values
        for j, filename in enumerate(filenames):
            ax.scatter(coords[j, 0], coords[j, 1], label=filename, s=150, edgecolors='black')
            ax.annotate(filename, (coords[j, 0], coords[j, 1]), xytext=(5, 5), textcoords='offset points')

        ax.set_title(f"Modello: {model_name.upper()}", fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("Componente Principale 1")
        ax.set_ylabel("Componente Principale 2")

    plt.suptitle("Analisi Identità Vocale - Confronto Modelli", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Crea cartella data se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Grafico salvato con successo in: {output_path}")
    plt.close()

def main():
    db_path = os.path.join("data", "embedding_db.pkl")
    data = load_embeddings(db_path)
    
    if data is None or len(data) == 0:
        print(f"❌ Errore: Database {db_path} non trovato o vuoto.")
        return

    # Trasformiamo in DataFrame
    df = pd.DataFrame(data)
    
    # Pulizia nomi file (togliamo .wav per il grafico)
    df['filename'] = df['filename'].apply(lambda x: os.path.basename(x))

    try:
        plot_voice_comparison(df)
    except Exception as e:
        print(f"❌ Errore critico durante la generazione del grafico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()