import os
import sys
import subprocess
import pickle
import numpy as np

# --- SETUP PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
# VOICE_DATA_DIR (impostata in run_analysis.sh) decide dove vengono letti gli
# embedding e scritto il database. Default: ./data accanto a questo script.
DATA_DIR = os.environ.get("VOICE_DATA_DIR", os.path.join(CURRENT_DIR, "data"))

def run_script(script_path):
    """Esegue uno script python come processo separato."""
    print(f"\n🚀 Esecuzione: {os.path.basename(script_path)}...")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"❌ Errore durante l'esecuzione di {script_path}")
        return False
    return True

def compile_database():
    """Raccoglie tutti i file .npy e crea il master database formattato in formato lungo (Long Format) per Pandas."""
    print("\n🗄️ Compilazione del Master Database in corso...")
    embeddings_dir = os.path.join(DATA_DIR, "embeddings")
    db_path = os.path.join(DATA_DIR, "embedding_db.pkl")
    
    rows = []
    
    if not os.path.exists(embeddings_dir):
        print("❌ Cartella embeddings non trovata!")
        return False

    models = [d for d in os.listdir(embeddings_dir) if os.path.isdir(os.path.join(embeddings_dir, d))]
    
    for model in models:
        model_dir = os.path.join(embeddings_dir, model)
        for file in os.listdir(model_dir):
            if file.endswith(".npy"):
                base_name = file.rsplit('_', 1)[0] + ".wav"
                file_path = os.path.join(model_dir, file)
                
                # Invece di raggruppare, creiamo una singola riga per ogni estrazione
                row = {
                    "filename": base_name,
                    "model": model,          # Aggiungiamo esplicitamente la colonna 'model'
                    "embedding": np.load(file_path) # Il vettore dati
                }
                rows.append(row)

    if not rows:
        print("⚠️ Nessun embedding trovato da compilare!")
        return False
        
    # Creiamo il DataFrame e lo salviamo
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_pickle(db_path)
        
    print(f"✅ Database compilato con successo! Salvato in: {db_path}")
    print(f"📊 Totale righe create: {len(df)}")
    return True

def main():
    print("🌟 AVVIO PIPELINE MULTIMEDIA PROJECT 🌟")
    
    # 1. Estrazione degli Embeddings
    super_wrapper = os.path.join(SRC_DIR, "super_wrapper.py")
    if not run_script(super_wrapper):
        sys.exit(1)
        
    # 1.5 Compilazione del Database (IL PEZZO MANCANTE!)
    if not compile_database():
        sys.exit(1)
        
    # 2. Analisi e Grafici
    total_embed = os.path.join(CURRENT_DIR, "total_embed.py")
    if os.path.exists(total_embed):
        if not run_script(total_embed):
            sys.exit(1)
    else:
        print(f"⚠️ Script di analisi {total_embed} non trovato.")

    print("\n✅ TUTTE LE OPERAZIONI SONO STATE COMPLETATE!")

if __name__ == "__main__":
    main()