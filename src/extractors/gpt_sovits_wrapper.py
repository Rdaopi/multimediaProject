import os
import sys
import torch
import librosa
import numpy as np
from transformers import AutoModel, AutoConfig

# Setup Percorsi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from base_encoder import BaseVoiceEncoder

class GPTSoVITSExtractor(BaseVoiceEncoder):
    def load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(PROJECT_ROOT, "checkpoints", "gpt_sovits", "pretrained_models")
        
        specific_weight_path = os.path.join(self.model_path, "chinese-hubert-base.bin")
        
        print(f"[GPT-SoVITS] Caricamento HuBERT da: {specific_weight_path}")
        config = AutoConfig.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(specific_weight_path, config=config)
        self.model.to(self.device).eval()

    def extract(self, audio_path):
        try:
            wav, _ = librosa.load(audio_path, sr=16000)
            wav_tensor = torch.from_numpy(wav).float().to(self.device).unsqueeze(0)

            with torch.no_grad():
                # Estrazione embedding standard
                outputs = self.model(wav_tensor)
                embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"❌ Errore: {e}")
            return None

if __name__ == "__main__":
    extractor = GPTSoVITSExtractor()
    extractor.process_all(os.path.join(PROJECT_ROOT, "data", "raw_vctk"), 
                          os.path.join(PROJECT_ROOT, "data", "embeddings", "gpt_sovits"), 
                          suffix="_gpt.npy")