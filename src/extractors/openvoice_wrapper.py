import os
import sys
import torch
import warnings
import numpy as np

# Percorsi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

# Setup modelli (verificati da setup_project.py)
OPENVOICE_PARENT = os.path.join(PROJECT_ROOT, "models_src", "OpenVoice")
sys.path.insert(0, OPENVOICE_PARENT)

from base_encoder import BaseVoiceEncoder
from openvoice.api import ToneColorConverter

warnings.filterwarnings("ignore")

class OpenVoiceExtractor(BaseVoiceEncoder):
    def load_model(self):
        self.device = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2")
        config_path = os.path.join(ckpt_dir, 'config.json')
        ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')

        print(f"[OpenVoice] Loading (Device: {self.device})")
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
    input_dir = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_dir = os.path.join(PROJECT_ROOT, "data", "embeddings", "openvoice")
    extractor.process_all(input_dir, output_dir, suffix="_ov.npy")