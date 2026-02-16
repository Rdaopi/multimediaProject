import os
import sys
import torch
import torch.nn as nn
import yaml
import librosa
import numpy as np
from torch.nn.utils import remove_spectral_norm

# Percorsi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

# Setup modelli (verificati da setup_project.py)
STYLE_PATH = os.path.join(PROJECT_ROOT, "models_src", "StyleTTS2")
sys.path.insert(0, STYLE_PATH)

from base_encoder import BaseVoiceEncoder
from models import StyleEncoder

# --- Utility Classes ---
class SqueezeWrapper(nn.Module):
    """Wrapper per correggere le dimensioni del tensore (Fix StyleTTS2)"""
    def __init__(self, original_layer):
        super().__init__()
        self.layer = original_layer
    
    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.dim() == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return self.layer(x)

class StyleTTS2Extractor(BaseVoiceEncoder):
    def load_model(self):
        self.device = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "styletts2_libri")
        config_path = os.path.join(ckpt_dir, "config.yml")
        
        # Carica Config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        m_params = config['model_params']
        self.sr = config['preprocess_params'].get('sr', 24000)

        # Checkpoint verificato da setup_project.py
        model_path = os.path.join(ckpt_dir, "epochs_2nd_00020.pth")

        print(f"[StyleTTS2] Init (Device: {self.device})")
        
        # Build Modello
        self.model = StyleEncoder(
            dim_in=m_params.get('dim_in', 64),
            style_dim=m_params.get('style_dim', 128),
            max_conv_dim=m_params.get('max_conv_dim', 512)
        ).to(self.device)
        
        self._apply_patches()
        self._load_weights(model_path)
        self.model.eval()

    def _apply_patches(self):
        """Applica remove_spectral_norm e SqueezeWrapper"""
        for name, module in self.model.named_modules():
            try:
                remove_spectral_norm(module)
            except ValueError: pass

        # Logica SqueezeWrapper
        if hasattr(self.model, 'unshared'):
            self.model.unshared = SqueezeWrapper(self.model.unshared)
        else:
            for name, module in self.model.named_children():
                if isinstance(module, nn.Linear) and module.in_features == 512:
                    setattr(self.model, name, SqueezeWrapper(module))
                    break

    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        full_net = checkpoint.get('net', checkpoint)
        encoder_dict = {}
        
        # Logica estrazione pesi specifica
        for k, v in full_net.items():
            if 'style_encoder' in k:
                encoder_dict[k.replace('style_encoder.', '')] = v
        # Se vuoto, prova a caricare tutto (magari è solo l'encoder salvato)
        if not encoder_dict: 
            encoder_dict = full_net

        self.model.load_state_dict(encoder_dict, strict=False)

    def extract(self, audio_path):
        try:
            wav, _ = librosa.load(audio_path, sr=self.sr)
            wav = wav / (np.max(np.abs(wav)) + 1e-7)
            
            mel = librosa.feature.melspectrogram(
                y=wav, sr=self.sr, n_mels=80, n_fft=2048, hop_length=300, win_length=1200
            )
            mel = np.log(np.clip(mel, a_min=1e-5, a_max=None)) 
            
            # Padding
            _, time_steps = mel.shape
            if time_steps % 32 != 0:
                pad_len = 32 - (time_steps % 32)
                mel = np.pad(mel, ((0, 0), (0, pad_len)), mode='constant')

            mel_tensor = torch.from_numpy(mel).float().to(self.device)
            if mel_tensor.ndim == 2: mel_tensor = mel_tensor.unsqueeze(0) 

            with torch.no_grad():
                emb = self.model(mel_tensor)
            
            return emb.cpu().numpy().squeeze()
        except Exception as e:
            print(f"❌ {os.path.basename(audio_path)}: {e}")
            return None

if __name__ == "__main__":
    extractor = StyleTTS2Extractor()
    input_dir = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_dir = os.path.join(PROJECT_ROOT, "data", "embeddings", "styletts2")
    extractor.process_all(input_dir, output_dir, suffix="_style.npy")