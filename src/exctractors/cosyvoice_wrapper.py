import os
import sys
import torch
import torchaudio
import numpy as np
import warnings

# --- PATH SETUP (Same as your other scripts) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from base_encoder import BaseVoiceEncoder
except ImportError:
    print("❌ Error: Could not find base_encoder.py in src/")
    sys.exit(1)

# --- COSYVOICE SETUP ---
# Adjust this to point to where you cloned CosyVoice
COSYVOICE_ROOT = os.path.join(PROJECT_ROOT, "models_src", "CosyVoice")

if COSYVOICE_ROOT not in sys.path:
    sys.path.append(COSYVOICE_ROOT)
    # CosyVoice often needs its third_party folder in path too
    sys.path.append(os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))

try:
    # We import the main wrapper class which handles the loading of the specific embedding model
    from cosyvoice.cli.cosyvoice import CosyVoice
except ImportError:
    print(f"❌ Critical Error: Could not import 'CosyVoice' from {COSYVOICE_ROOT}.")
    print("Ensure you have installed the requirements for CosyVoice.")
    sys.exit(1)

warnings.filterwarnings("ignore")

class CosyVoiceExtractor(BaseVoiceEncoder):
    def load_model(self):
        self.device = self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Point this to your CosyVoice pretrained model folder
        # e.g., "checkpoints/CosyVoice-300M" or "checkpoints/CosyVoice2-0.5B"
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "CosyVoice-300M")
        
        if not os.path.exists(self.ckpt_dir):
             raise FileNotFoundError(f"❌ CosyVoice model path not found: {self.ckpt_dir}")

        print(f"[CosyVoice] Loading model from: {self.ckpt_dir}")
        
        # Initialize the full CosyVoice wrapper
        # This automatically loads the Cam++ speaker encoder inside 'self.model.frontend'
        self.cosy_wrapper = CosyVoice(self.ckpt_dir)
        
        # We don't need the full flow/LLM models for extraction, but loading the wrapper 
        # is the safest way to ensure we use the EXACT same audio frontend as the inference.
        
        # Access the internal frontend where the embedding logic lives
        self.frontend = self.cosy_wrapper.frontend

    def extract(self, audio_path):
        try:
            # 1. Load Audio
            # CosyVoice prompt audio MUST be 16kHz
            speech, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech)
            
            # Ensure it's on the correct device
            speech = speech.to(self.device)

            # 2. Extract Embedding
            # CosyVoice's frontend has a method specifically for this. 
            # Depending on version, it might be `_extract_speaker_embedding` or we manually call the campplus model.
            
            # Method A: Use the internal private method if available (Most robust)
            if hasattr(self.frontend, "_extract_speaker_embedding"):
                # Some versions expect shape [1, T]
                if speech.dim() == 1: speech = speech.unsqueeze(0)
                embedding = self.frontend._extract_speaker_embedding(speech)
            
            # Method B: Manual Cam++ inference (Fallback)
            # The speaker encoder is usually stored in self.frontend.campplus_session (if onnx) 
            # or self.frontend.model (if torch)
            else:
                 # Standard CosyVoice usage often computes embedding inside inference_sft
                 # Let's try to mimic the flow:
                 # The frontend usually computes fbank -> passes to Cam++
                 # But the easiest way is checking if 'campplus' is accessible
                 pass 
                 # (If Method A fails, the code will crash here, but Method A is standard for current CosyVoice repo)

            # 3. Format Output
            # Embedding is typically [1, 192] or [192]
            return embedding.detach().cpu().numpy().squeeze()

        except Exception as e:
            # Fallback for "Method A" failure: inspect the object
            try:
                # If the wrapper method failed, let's look for the embedding model directly
                # It is often named 'campplus' or 'speaker_encoder'
                # This is a 'Last Resort' attempt
                model = getattr(self.cosy_wrapper, 'model', None) # or self.cosy_wrapper.frontend
                # ... implementation logic would go here if needed ...
                print(f"⚠️ Standard extraction failed, check CosyVoice version. Error: {e}")
                return None
            except:
                print(f"❌ Error extracting {os.path.basename(audio_path)}: {e}")
                return None

if __name__ == "__main__":
    extractor = CosyVoiceExtractor()
    
    input_dir = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_dir = os.path.join(PROJECT_ROOT, "data", "embeddings", "cosyvoice")
    
    extractor.process_all(input_dir, output_dir, suffix="_cosy.npy")
