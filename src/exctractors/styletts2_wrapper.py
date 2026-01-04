import os
import sys
import torch
import yaml
import librosa
import numpy as np

# Percorsi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
STYLE_PATH = os.path.join(PROJECT_ROOT, "models_src", "StyleTTS2")
sys.path.append(STYLE_PATH)

# Importiamo moduli interni di StyleTTS2
try:
    from models import build_model
    # Proviamo a importare utilità per l'audio se presenti, altrimenti useremo librosa custom
    from utils import load_checkpoint
except ImportError:
    raise ImportError(f"Non trovo i moduli in {STYLE_PATH}. Hai fatto git clone di StyleTTS2?")

class StyleTTS2Extractor:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "styletts2")
        self.config_path = os.path.join(self.ckpt_dir, "config.yml")
        self.model_path = os.path.join(self.ckpt_dir, "Epoch_0630.pth")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Manca Epoch_0630.pth in {self.ckpt_dir}")

        # 1. Carichiamo la configurazione
        config = yaml.safe_load(open(self.config_path))
        
        # 2. Istanziamo il modello (solo la parte necessaria)
        # Usiamo un trucco: passiamo un oggetto fittizio come "args" se richiesto, o usiamo il dizionario
        # StyleTTS2 usa F0_path, ASIC, ecc. nel config. 
        self.model_params = config['model_params']
        
        print(f"[StyleTTS2] Inizializzazione modello...")
        # build_model è una funzione di StyleTTS2 che costruisce la rete
        self.model = build_model(self.model_params, load_checkpoint=False).to(self.device)
        
        # 3. Carichiamo i pesi
        params = torch.load(self.model_path, map_location='cpu')
        params = params['net'] if 'net' in params else params
        
        # Carichiamo gestendo eventuali prefissi 'module.' (comune se trainato su multi-GPU)
        new_state_dict = {}
        for k, v in params.items():
            key = k[7:] if k.startswith('module.') else k
            new_state_dict[key] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        print("[StyleTTS2] Modello caricato.")

        # Parametri per lo spettrogramma (devono coincidere col training)
        self.sr = config['preprocess_params'].get('sr', 24000)
    
    def preprocess_audio(self, audio_path):
        """Converte audio -> Mel Spectrogram come si aspetta StyleTTS2"""
        # Carica audio e resample
        wav, _ = librosa.load(audio_path, sr=self.sr)
        wav = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        
        # Calcolo Mel Spectrogram usando le utility interne del modello sarebbe l'ideale,
        # ma spesso è complesso importarle. 
        # Qui sfruttiamo il fatto che lo style_encoder prende in input le feature
        # o chiamiamo direttamente il metodo interno se disponibile.
        
        # NOTA: StyleTTS2 ha un modulo `preprocess` interno ma per semplicità
        # qui assumiamo di poter passare l'audio grezzo se il modello ha un frontend, 
        # ALTRIMENTI dobbiamo calcolare lo spec. 
        
        # Approccio più sicuro per StyleTTS2: computare mel usando librosa coi parametri del config
        # Ma per ora proviamo a usare la funzione interna dello style encoder se accetta raw wav?
        # No, lo style encoder vuole mel.
        
        # Usiamo torchaudio per replicare il preprocess del config
        import torchaudio
        to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=2048, win_length=1200, hop_length=300, n_mels=80
        ).to(self.device)
        
        mel = to_mel(wav)
        
        # Logarithmic compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

    def extract(self, audio_path):
        if not os.path.exists(audio_path):
            return None
            
        try:
            # Prepara il mel spectrogram
            mel_tensor = self.preprocess_audio(audio_path)
            
            # Passa allo style encoder
            # Il modello StyleTTS2 ha solitamente self.model.style_encoder
            with torch.no_grad():
                # L'input dello style encoder è (batch, mels, time)
                style_emb = self.model.style_encoder(mel_tensor)
                
            return style_emb.detach().cpu().numpy().squeeze()
            
        except Exception as e:
            print(f"Errore StyleTTS2 su {audio_path}: {e}")
            return None

if __name__ == "__main__":
    extractor = StyleTTS2Extractor()
    dummy_audio = os.path.join(PROJECT_ROOT, "data", "test_sample.wav")
    if os.path.exists(dummy_audio):
        emb = extractor.extract(dummy_audio)
        print(f"StyleTTS2 embedding shape: {emb.shape}")