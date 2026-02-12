import os
import sys
import torch
import gc
import time

# --- SETUP PERCORSI ---
# Posizione corrente: PROJECT_ROOT/src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Root del progetto: PROJECT_ROOT/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# Cartella degli estrattori: PROJECT_ROOT/src/extractors/
EXTRACTORS_DIR = os.path.join(CURRENT_DIR, "extractors")

# Aggiungiamo src e extractors al path di sistema per Python
if CURRENT_DIR not in sys.path: sys.path.append(CURRENT_DIR)
if EXTRACTORS_DIR not in sys.path: sys.path.append(EXTRACTORS_DIR)

from cosyvoice_wrapper import CosyVoiceExtractor
from gpt_sovits_wrapper import GPTSoVITSExtractor
from openvoice_wrapper import OpenVoiceExtractor
from styletts2_wrapper import StyleTTS2Extractor

def clean_memory():
    """Forza la pulizia della VRAM e della RAM tra un modello e l'altro."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(2) # Pausa leggermente più lunga per sicurezza

def main():
    # --- CONFIGURAZIONE ---
    # Input: File WAV originali
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    
    # Output: Cartella base per gli embeddings
    OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data", "embeddings")

    # Lista Tasks: (Classe, nome_cartella_output, suffisso_file)
    tasks = [
        (CosyVoiceExtractor, "cosyvoice",  "_cosy.npy"),
        (GPTSoVITSExtractor, "gpt_sovits", "_gpt.npy"),
        (OpenVoiceExtractor, "openvoice",  "_ov.npy"),
        (StyleTTS2Extractor, "styletts2",  "_style.npy"),
    ]

    print(f"🌟 SUPER WRAPPER (Master) AVVIATO")
    print(f"📍 Esecuzione da: {CURRENT_DIR}")
    print(f"📂 Input Data: {INPUT_DIR}")
    print("--------------------------------------------------")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Errore critico: Cartella input non trovata: {INPUT_DIR}")
        return

    # --- ESECUZIONE SEQUENZIALE ---
    for ExtractorClass, sub_dir, suffix in tasks:
        model_name = ExtractorClass.__name__
        target_dir = os.path.join(OUTPUT_BASE, sub_dir)
        
        print(f"\n🔄 [Avvio Modulo] {model_name}")
        
        extractor = None
        try:
            # 1. Caricamento Modello (Inizializza la classe e carica i pesi in VRAM)
            extractor = ExtractorClass()
            
            # 2. Elaborazione (Loop sui file ereditato da BaseVoiceEncoder)
            print(f"▶️ Elaborazione in corso verso: {target_dir}")
            extractor.process_all(INPUT_DIR, target_dir, suffix=suffix)
            
        except Exception as e:
            print(f"⚠️ ERRORE CRITICO in {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 3. Pulizia Totale (Essenziale per evitare Out Of Memory)
            if extractor:
                del extractor
            print(f"🧹 Pulizia memoria VRAM post-{model_name}...")
            clean_memory()
            print("--------------------------------------------------")

    print("\n✅ PIPELINE COMPLETATA CON SUCCESSO!")

if __name__ == "__main__":
    main()