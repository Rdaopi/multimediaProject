import os
import sys
import torch
import numpy as np

# 1. SETUP DEI PERCORSI
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

# La cartella che CONTIENE 'openvoice'
OPENVOICE_PARENT = os.path.join(PROJECT_ROOT, "models_src", "OpenVoice")
# La cartella effettiva del codice
OPENVOICE_SRC = os.path.join(OPENVOICE_PARENT, "openvoice")

# Aggiungiamo ENTRAMBE al sistema
if OPENVOICE_PARENT not in sys.path:
    sys.path.append(OPENVOICE_PARENT)
if OPENVOICE_SRC not in sys.path:
    sys.path.append(OPENVOICE_SRC)

try:
    from api import ToneColorConverter
    print("✅ Modulo 'api' e dipendenze interne caricati!")
except Exception as e:
    print(f"❌ Errore durante l'import: {e}")
    sys.exit(1)
    
class OpenVoiceExtractor:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Percorsi dei pesi (hardcoded per semplicità, ma modificabili)
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2")
        config_path = os.path.join(self.ckpt_dir, 'config.json')
        ckpt_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Mancano i pesi in: {ckpt_path}")

        print(f"[OpenVoice] Caricamento modello su {self.device}...")
        self.converter = ToneColorConverter(config_path, device=self.device)
        self.converter.load_ckpt(ckpt_path)
        print("[OpenVoice] Pronto.")

    def extract(self, audio_path):
        """
        Input: path del file audio (.wav)
        Output: numpy array del vettore embedding
        """
        if not os.path.exists(audio_path):
            print(f"File non trovato: {audio_path}")
            return None
            
        # extract_se ritorna il target_se (l'embedding che cerchiamo)
        try:
            target_se, _ = self.converter.extract_se(audio_path, se_save_path=None)
            # Convertiamo in numpy e appiattiamo eventuali dimensioni extra [1, 256, 1] -> [256]
            return target_se.detach().cpu().numpy().squeeze()
        except Exception as e:
            print(f"Errore estrazione OpenVoice su {audio_path}: {e}")
            return None

# Test rapido se esegui questo file direttamente
if __name__ == "__main__":
    extractor = OpenVoiceExtractor()
    # Cambia con un tuo file audio reale per testare
    dummy_audio = os.path.join(PROJECT_ROOT, "data", "test_sample.wav") 
    if os.path.exists(dummy_audio):
        emb = extractor.extract(dummy_audio)
        print(f"Shape estratta: {emb.shape}")