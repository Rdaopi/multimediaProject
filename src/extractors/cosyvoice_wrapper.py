import os
import sys
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort
import numpy as np
import librosa  # <--- USIAMO QUESTO CHE FUNZIONA!

# --- SETUP PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)

try:
    from base_encoder import BaseVoiceEncoder
except ImportError:
    print("❌ Errore: base_encoder.py non trovato.")
    sys.exit(1)

class CosyVoiceExtractor(BaseVoiceEncoder):
    def load_model(self):
        """Carica il modello ONNX."""
        # Percorso del modello
        self.onnx_path = os.path.join(PROJECT_ROOT, "models_src", "CosyVoice", "pretrained_models", "CosyVoice-300M", "campplus.onnx")
        
        # Fallback
        if not os.path.exists(self.onnx_path):
             self.onnx_path = os.path.join(PROJECT_ROOT, "checkpoints", "CosyVoice-300M", "campplus.onnx")

        if not os.path.exists(self.onnx_path):
             raise FileNotFoundError(f"❌ Modello non trovato!\nCercato in: {self.onnx_path}")

        print(f"[CosyVoice] Loading Cam++ (ONNX) da: {self.onnx_path}")
        
        # Sessione ONNX (Gestione errori CUDA)
        try:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3 
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, sess_options=sess_options, providers=providers)
            print(f"[CosyVoice] Sessione avviata (Provider: {self.session.get_providers()[0]})")
        except Exception as e:
            print(f"⚠️ GPU non disponibile, uso CPU.")
            self.session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])

    def _compute_fbank(self, audio_path):
        """
        Versione ROBUSTA: Usa Librosa per caricare (funziona sempre su Windows)
        e Torchaudio per i calcoli Fbank.
        """
        # 1. Carica con Librosa direttamente a 16kHz (bypassiamo i codec di Torch)
        wav_numpy, _ = librosa.load(audio_path, sr=16000)
        
        # 2. Converti in Tensore Torch [1, T]
        wav = torch.from_numpy(wav_numpy).float()
        wav = wav.unsqueeze(0) # Aggiungi dimensione batch -> [1, T]
        
        # 3. Calcolo Fbank (Matematica Kaldi)
        # Nota: librosa carica in float -1..1, Kaldi vuole PCM 16bit scalato
        wav = wav * (1 << 15) 
        feat = kaldi.fbank(wav, num_mel_bins=80, 
                           frame_length=25, frame_shift=10, 
                           dither=0.0, energy_floor=0.0,
                           sample_frequency=16000)
        
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat.unsqueeze(0).numpy() # [1, Time, 80]

    def extract(self, audio_path):
        try:
            # 1. Prepara audio
            fbank = self._compute_fbank(audio_path)
            
            # 2. Inferenza ONNX
            input_name = self.session.get_inputs()[0].name
            embedding = self.session.run(None, {input_name: fbank})[0]
            
            # 3. Output pulito
            return embedding.squeeze()

        except Exception as e:
            print(f"❌ Errore su {os.path.basename(audio_path)}: {e}")
            return None

if __name__ == "__main__":
    extractor = CosyVoiceExtractor()
    
    # Percorsi corretti
    input_dir = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_dir = os.path.join(PROJECT_ROOT, "data", "embeddings", "cosyvoice")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Inizio estrazione (CosyVoice) da: {input_dir}")
    extractor.process_all(input_dir, output_dir, suffix="_cosy.npy")
