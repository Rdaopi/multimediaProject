import os
import sys
import torch
import torch.nn as nn
import yaml
import librosa
import numpy as np
from torch.nn.utils import remove_spectral_norm

# --- 1. SETUP PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
STYLE_PATH = os.path.join(PROJECT_ROOT, "models_src", "StyleTTS2")
sys.path.append(STYLE_PATH)

try:
    from models import StyleEncoder
except ImportError:
    print(f"❌ Errore: Impossibile importare StyleEncoder da {STYLE_PATH}")
    sys.exit(1)

# --- CLASSE ADATTATORE UNIVERSALE ---
class SqueezeWrapper(nn.Module):
    """
    Intercetta l'input. Se trova dimensioni inutili alla fine (es. 512x1), 
    le schiaccia prima di passare al livello Lineare.
    """
    def __init__(self, original_layer):
        super().__init__()
        self.layer = original_layer
    
    def forward(self, x):
        # DEBUG: Decommenta se vuoi vedere cosa passa
        # print(f"DEBUG LAYER: Input shape {x.shape}")
        
        # Caso 1: [Batch, 512, 1] -> [Batch, 512]
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            
        # Caso 2: [512, 1] -> [512] (Il caso che ci fregava prima!)
        elif x.dim() == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            
        # Caso Sicurezza: Se dopo lo squeeze è [512], PyTorch Linear lo gestisce.
        # Se fosse [1, 512], lo gestisce. 
        # L'importante è togliere quell'ultima dimensione se è 1.
        
        return self.layer(x)

class StyleTTS2Extractor:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "styletts2_libri")
        config_path = os.path.join(self.ckpt_dir, "config.yml")
        
        # Trova file .pth
        pth_files = [f for f in os.listdir(self.ckpt_dir) if f.endswith('.pth')]
        self.model_path = os.path.join(self.ckpt_dir, pth_files[0])

        # Config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        m_params = config['model_params']
        self.sr = config['preprocess_params'].get('sr', 24000)

        print(f"[StyleTTS2] Inizializzazione su {self.device}...")
        
        # 1. Istanziamo il modello
        self.model = StyleEncoder(
            dim_in=m_params.get('dim_in', 64),
            style_dim=m_params.get('style_dim', 128),
            max_conv_dim=m_params.get('max_conv_dim', 512)
        ).to(self.device)
        
        # 2. RIMUOVIAMO SPECTRAL NORM
        for name, module in self.model.named_modules():
            try:
                remove_spectral_norm(module)
            except ValueError: pass

        # 3. APPLICAZIONE AGGRESSIVA DEL FIX
        # Cerchiamo ricorsivamente il layer 'unshared' o qualsiasi Linear compatibile
        applied_fix = False
        
        # Tentativo A: Nome standard
        if hasattr(self.model, 'unshared'):
            print("🔧 Trovato layer 'unshared', applico wrapper...")
            self.model.unshared = SqueezeWrapper(self.model.unshared)
            applied_fix = True
        else:
            # Tentativo B: Ricerca bruta (se il nome fosse diverso)
            print("⚠️ Layer 'unshared' non trovato per nome. Cerco Linear(512, 128)...")
            for name, module in self.model.named_children():
                if isinstance(module, nn.Linear) and module.in_features == 512:
                    print(f"🔧 Trovato layer candidato '{name}', applico wrapper...")
                    setattr(self.model, name, SqueezeWrapper(module))
                    applied_fix = True
                    break
        
        if not applied_fix:
            print("⚠️ ATTENZIONE: Non sono riuscito ad applicare il fix SqueezeWrapper!")

        # 4. CARICAMENTO PESI
        print(f"📂 Caricamento pesi da: {os.path.basename(self.model_path)}")
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        full_net = checkpoint.get('net', checkpoint)
        
        if 'style_encoder' in full_net:
            encoder_dict = full_net['style_encoder']
        else:
            encoder_dict = {}
            for k, v in full_net.items():
                if 'style_encoder' in k:
                    encoder_dict[k.replace('style_encoder.', '')] = v

        try:
            self.model.load_state_dict(encoder_dict, strict=False)
            print(f"✅ Pesi caricati correttamente.")
        except RuntimeError as e:
            print(f"❌ Errore caricamento: {e}")
            sys.exit(1)
        
        self.model.eval()

    def extract(self, audio_path):
        try:
            # Pre-processing Audio
            wav, _ = librosa.load(audio_path, sr=self.sr)
            wav = wav / (np.max(np.abs(wav)) + 1e-7)
            
            mel = librosa.feature.melspectrogram(
                y=wav, sr=self.sr, n_mels=80, n_fft=2048, hop_length=300, win_length=1200
            )
            mel = np.log(np.clip(mel, a_min=1e-5, a_max=None)) 
            
            # --- FIX PADDING ---
            _, time_steps = mel.shape
            if time_steps % 32 != 0:
                pad_len = 32 - (time_steps % 32)
                mel = np.pad(mel, ((0, 0), (0, pad_len)), mode='constant')

            # Formato tensore
            mel_tensor = torch.from_numpy(mel).float().to(self.device)
            if mel_tensor.ndim == 2:
                mel_tensor = mel_tensor.unsqueeze(0) 

            with torch.no_grad():
                emb = self.model(mel_tensor)
            
            return emb.cpu().numpy().squeeze()
            
        except Exception as e:
            print(f"❌ Errore su {os.path.basename(audio_path)}: {e}")
            # Importante: stampa lo stack trace se serve capire dove si rompe
            # import traceback
            # traceback.print_exc()
            return None

    def process_all(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        
        print(f"🚀 Batch StyleTTS2: {len(files)} file...")
        successi = 0
        for file in files:
            emb = self.extract(os.path.join(input_dir, file))
            if emb is not None:
                np.save(os.path.join(output_dir, file.replace('.wav', '_style.npy')), emb)
                successi += 1
                print(f"   [OK] {file}")
        
        print(f"📊 Finito: {successi}/{len(files)}")

if __name__ == "__main__":
    extractor = StyleTTS2Extractor()
    input_vctk = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_style = os.path.join(PROJECT_ROOT, "data", "embeddings", "styletts2")
    
    if os.path.exists(input_vctk):
        extractor.process_all(input_vctk, output_style)