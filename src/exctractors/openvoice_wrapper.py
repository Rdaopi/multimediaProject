import os
import sys
import torch
import numpy as np
import warnings

# Silenziamo i warning per un output pulito
warnings.filterwarnings("ignore")

# --- 1. SETUP DEI PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

OPENVOICE_PARENT = os.path.join(PROJECT_ROOT, "models_src", "OpenVoice")
OPENVOICE_SRC = os.path.join(OPENVOICE_PARENT, "openvoice")

if OPENVOICE_PARENT not in sys.path:
    sys.path.append(OPENVOICE_PARENT)
if OPENVOICE_SRC not in sys.path:
    sys.path.append(OPENVOICE_SRC)

try:
    from api import ToneColorConverter
except ImportError:
    print(f"❌ Errore critico: Impossibile importare 'api' da OpenVoice.")
    sys.exit(1)

# --- 2. CLASSE ESTRATTORE ---
class OpenVoiceExtractor:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2")
        config_path = os.path.join(self.ckpt_dir, 'config.json')
        ckpt_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ Pesi non trovati in: {ckpt_path}")

        print(f"[OpenVoice] Inizializzazione su: {self.device}")
        self.converter = ToneColorConverter(config_path, device=self.device)
        self.converter.load_ckpt(ckpt_path)
        print("[OpenVoice] Modello pronto.")

    def extract(self, audio_path):
        """Estrae l'embedding gestendo il possibile output singolo della V2"""
        try:
            # Correzione errore unpacking: gestiamo sia ritorno singolo che tupla
            result = self.converter.extract_se(audio_path, se_save_path=None)
            target_se = result[0] if isinstance(result, (tuple, list)) else result
            return target_se.detach().cpu().numpy().squeeze()
        except Exception as e:
            print(f"   ❌ Errore su {os.path.basename(audio_path)}: {e}")
            return None

    def process_all(self, input_dir, output_dir):
        """Processa tutti i file .wav trovati"""
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        
        if not files:
            print(f"⚠️ Nessun file .wav in {input_dir}")
            return

        print(f"🚀 Inizio estrazione batch per {len(files)} file...")
        successi = 0
        
        for file in files:
            full_path = os.path.join(input_dir, file)
            emb = self.extract(full_path)
            
            if emb is not None:
                out_name = os.path.splitext(file)[0] + "_ov.npy"
                np.save(os.path.join(output_dir, out_name), emb)
                successi += 1
                print(f"   [OK] {file} -> {out_name}")

        print(f"\n✅ Elaborazione completata!")
        print(f"📊 File processati con successo: {successi}/{len(files)}")
        print(f"📂 Cartella output: {output_dir}")

# --- 3. ESECUZIONE ---
if __name__ == "__main__":
    extractor = OpenVoiceExtractor()
    
    # Cartelle basate sulla tua struttura
    input_vctk = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    output_ov = os.path.join(PROJECT_ROOT, "data", "embeddings", "openvoice")
    
    if os.path.exists(input_vctk):
        extractor.process_all(input_vctk, output_ov)
    else:
        print(f"❌ Cartella non trovata: {input_vctk}")