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
            # 1. Preprocessing: GPT-SoVITS richiede tassativamente 16kHz per l'encoder SSL
            wav_16k, _ = librosa.load(audio_path, sr=16000)
            
            # Normalizzazione semplice (spesso utile per SSL)
            wav_16k = wav_16k / (np.max(np.abs(wav_16k)) + 1e-7)

            # Preparazione tensore [1, T]
            wav_tensor = torch.from_numpy(wav_16k).float().to(self.device)
            wav_tensor = wav_tensor.unsqueeze(0)

            with torch.no_grad():
                # 2. Estrazione Feature
                # Il modello restituisce un dizionario o una tupla, noi vogliamo 'last_hidden_state'
                # cnhubert.get_model() restituisce un modello wav2vec2 modificato
                result = self.model(wav_tensor, output_hidden_states=True)
                
                # In GPT-SoVITS si usa solitamente l'output del 12° layer o l'ultimo stato
                # result['last_hidden_state'] ha shape [Batch, Frames, 768]
                if hasattr(result, 'last_hidden_state'):
                    features = result.last_hidden_state
                else:
                    # Fallback per versioni diverse di transformers/hubert
                    features = result[0]

            # 3. Mean Pooling per ottenere un vettore 1D (Impronta Vocale)
            # Trasformiamo la sequenza temporale in un singolo vettore facendo la media
            embedding = torch.mean(features, dim=1).squeeze().cpu().numpy()
            
            return embedding

        except Exception as e:
            print(f"❌ Errore {os.path.basename(audio_path)}: {e}")
            return None

if __name__ == "__main__":
    extractor = GPTSoVITSExtractor()
    
    # Percorsi dati (adatta se necessario)
    input_vctk = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_gpt = os.path.join(PROJECT_ROOT, "data", "embeddings", "gpt_sovits")
    
    # Chiamata batch
    extractor.process_all(input_vctk, output_gpt, suffix="_gpt.npy")
