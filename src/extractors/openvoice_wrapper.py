import os
import sys
import torch
import warnings
import numpy as np

# --- SETUP PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Definiamo la cartella src (un livello sopra extractors)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

# Aggiungiamo src al path per poter importare base_encoder
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Importiamo la classe base
try:
    from base_encoder import BaseVoiceEncoder
except ImportError:
    print("❌ Errore: Non trovo base_encoder.py in src/")
    sys.exit(1)

# Percorsi specifici OpenVoice
OPENVOICE_PARENT = os.path.join(PROJECT_ROOT, "models_src", "OpenVoice")
OPENVOICE_SRC = os.path.join(OPENVOICE_PARENT, "openvoice")

if OPENVOICE_PARENT not in sys.path: sys.path.append(OPENVOICE_PARENT)
if OPENVOICE_SRC not in sys.path: sys.path.append(OPENVOICE_SRC)

try:
    from api import ToneColorConverter
except ImportError:
    print(f"❌ Errore critico: Impossibile importare 'api' da OpenVoice.")
    sys.exit(1)

warnings.filterwarnings("ignore")

class OpenVoiceExtractor(BaseVoiceEncoder):
    def load_model(self):
        self.device = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2")
        config_path = os.path.join(ckpt_dir, 'config.json')
        ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ Pesi non trovati in: {ckpt_path}")

        print(f"[OpenVoice] Loading su: {self.device}")
        self.model = ToneColorConverter(config_path, device=self.device)
        self.model.load_ckpt(ckpt_path)

    def extract(self, audio_path):
        try:
            result = self.model.extract_se(audio_path, se_save_path=None)
            # Gestione tupla/lista vs tensore singolo
            target_se = result[0] if isinstance(result, (tuple, list)) else result
            return target_se.detach().cpu().numpy().squeeze()
        except Exception as e:
            print(f"❌ Errore {os.path.basename(audio_path)}: {e}")
            return None

if __name__ == "__main__":
    extractor = OpenVoiceExtractor()
    input_vctk = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_ov = os.path.join(PROJECT_ROOT, "data", "embeddings", "openvoice")
    
    # Chiamata al metodo ereditato dalla Base Class
    extractor.process_all(input_vctk, output_ov, suffix="_ov.npy")