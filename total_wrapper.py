import os
import sys
import subprocess

# --- SETUP PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumiamo che total_wrapper sia nella root (come da tuo layout proposto)
SRC_DIR = os.path.join(CURRENT_DIR, "src")

def run_script(script_path):
    """Esegue uno script python come processo separato."""
    print(f"\n🚀 Esecuzione: {os.path.basename(script_path)}...")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"❌ Errore durante l'esecuzione di {script_path}")
        return False
    return True

def main():
    print("🌟 AVVIO PIPELINE MULTIMEDIA PROJECT 🌟")
    
    # 1. Estrazione degli Embeddings
    # Puntiamo al super_wrapper in src/
    super_wrapper = os.path.join(SRC_DIR, "super_wrapper.py")
    if not run_script(super_wrapper):
        sys.exit(1)
        
    # 2. Analisi e Grafici
    # Puntiamo a total_embed nella root
    total_embed = os.path.join(CURRENT_DIR, "total_embed.py")
    if os.path.exists(total_embed):
        if not run_script(total_embed):
            sys.exit(1)
    else:
        print(f"⚠️ Script di analisi {total_embed} non trovato.")

    print("\n✅ TUTTE LE OPERAZIONI SONO STATE COMPLETATE!")

if __name__ == "__main__":
    main()