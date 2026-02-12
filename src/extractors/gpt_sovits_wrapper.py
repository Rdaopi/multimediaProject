import os
import sys
import torch
import librosa
import numpy as np
import warnings

# --- SETUP PERCORSI (Identico ai tuoi script) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from base_encoder import BaseVoiceEncoder
except ImportError:
    print("❌ Errore: Non trovo base_encoder.py in src/")
    sys.exit(1)

# --- SETUP GPT-SoVITS ---
# Assumiamo che la repo GPT-SoVITS sia in models_src/GPT-SoVITS
GPT_SOVITS_ROOT = os.path.join(PROJECT_ROOT, "models_src", "GPT-SoVITS")

# In GPT-SoVITS, il modulo feature_extractor è spesso dentro la cartella "GPT_SoVITS"
# Aggiungiamo entrambi i percorsi per sicurezza
GPT_SOVITS_SRC = os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS")

if GPT_SOVITS_ROOT not in sys.path: sys.path.append(GPT_SOVITS_ROOT)
if GPT_SOVITS_SRC not in sys.path: sys.path.append(GPT_SOVITS_SRC)

try:
    # Importiamo cnhubert dalla repo di GPT-SoVITS
    from feature_extractor import cnhubert
except ImportError:
    print(f"❌ Errore critico: Impossibile importare 'feature_extractor.cnhubert'.")
    print(f"Verifica che {GPT_SOVITS_SRC} esista e contenga feature_extractor/")
    sys.exit(1)

warnings.filterwarnings("ignore")

class GPTSoVITSExtractor(BaseVoiceEncoder):
    def load_model(self):
        self.device = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Percorso del modello SSL pre-addestrato (Chinese HuBERT)
        # Assumiamo che tu abbia scaricato i pesi in checkpoints/gpt_sovits/pretrained_models/
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "gpt_sovits", "pretrained_models")
        
        # Cerca il file del modello cnhubert
        cnhubert_base_path = os.path.join(self.ckpt_dir, "chinese-hubert-base")
        if not os.path.exists(cnhubert_base_path):
             # Fallback: a volte è direttamente un file .pt o .pth
             cnhubert_base_path = os.path.join(self.ckpt_dir, "chinese-hubert-base.pt")
             if not os.path.exists(cnhubert_base_path):
                raise FileNotFoundError(f"❌ Pesi CN-HuBERT non trovati in: {self.ckpt_dir}")

        print(f"[GPT-SoVITS] Loading CN-HuBERT da: {cnhubert_base_path}")
        print(f"[GPT-SoVITS] Device: {self.device}")

        # Setup del percorso per cnhubert (richiesto dalla sua logica interna)
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.model = cnhubert.get_model()
        self.model.eval()
        self.model.to(self.device)

    def extract(self, audio_path):
        try:
            # 1. Caricamento e Resample a 16kHz
            # librosa restituisce wav come array numpy 1D (T,)
            wav, _ = librosa.load(audio_path, sr=16000)
            wav = wav / (np.max(np.abs(wav)) + 1e-7) # Normalize
            
            # --- SOLUZIONE DEFINITIVA ---
            # 1. Convertiamo in Tensore (per soddisfare .device)
            # 2. NON aggiungiamo dimensioni (niente unsqueeze). Lasciamo che sia (T,)
            # 3. Spostiamo sul device corretto
            wav_tensor = torch.from_numpy(wav).float().to(self.device)

            with torch.no_grad():
                # Passiamo il tensore 1D. 
                # Il wrapper interno farà: feature_extractor(x) -> aggiunge batch -> (1, T) -> Conv1D felice.
                result = self.model(wav_tensor)
                
                # Gestione output
                if isinstance(result, torch.Tensor):
                    features = result
                elif isinstance(result, dict) and 'last_hidden_state' in result:
                    features = result['last_hidden_state']
                else:
                    features = result[0] if isinstance(result, (tuple, list)) else result

                # 3. Mean Pooling
                # features sarà [1, Time, 768]
                if features.dim() == 3:
                    embedding = torch.mean(features, dim=1).squeeze().cpu().numpy()
                else:
                    embedding = features.squeeze().cpu().numpy()
            
            return embedding

        except Exception as e:
            print(f"❌ Errore su {os.path.basename(audio_path)}: {e}")
            return None

if __name__ == "__main__":
    extractor = GPTSoVITSExtractor()
    
    # Percorsi dati (adatta se necessario)
    input_vctk = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_gpt = os.path.join(PROJECT_ROOT, "data", "embeddings", "gpt_sovits")
    
    # Chiamata batch
    extractor.process_all(input_vctk, output_gpt, suffix="_gpt.npy")
