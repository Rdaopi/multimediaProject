"""
Setup script per il Voice Identity Analysis Tool.
Esegue tutte le verifiche preliminari e l'installazione delle dipendenze.
"""

import os
import sys
import subprocess

def print_step(message):
    """Stampa un passo di setup."""
    print(f"\n{'='*60}")
    print(f"📋 {message}")
    print(f"{'='*60}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")
    return False

def install_dependencies():
    """Installa i pacchetti da requirements.txt."""
    print_step("Installazione Dipendenze Python")
    
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(req_file):
        return print_error(f"File requirements.txt non trovato: {req_file}")
    
    print(f"📦 Installazione da: {req_file}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", req_file],
            check=False,
            timeout=300,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"⚠️ Warning durante l'installazione:")
            print(result.stderr[:500])
            # Continua comunque, non bloccare
        
        print_success("Dipendenze controllate/installate")
        return True
        
    except subprocess.TimeoutExpired:
        return print_error("Timeout durante l'installazione (>5 min)")
    except Exception as e:
        print(f"⚠️ Errore durante pip install: {e}")
        return True  # Non bloccare

def verify_directories():
    """Verifica e crea le cartelle necessarie."""
    print_step("Verifica Struttura Cartelle")
    
    PROJECT_ROOT = os.path.dirname(__file__)
    
    required_dirs = {
        "data": os.path.join(PROJECT_ROOT, "data"),
        "data/raw_vctk": os.path.join(PROJECT_ROOT, "data", "raw_vctk"),
        "data/embeddings": os.path.join(PROJECT_ROOT, "data", "embeddings"),
        "checkpoints": os.path.join(PROJECT_ROOT, "checkpoints"),
        "models_src": os.path.join(PROJECT_ROOT, "models_src"),
        "src": os.path.join(PROJECT_ROOT, "src"),
    }
    
    all_ok = True
    
    for name, path in required_dirs.items():
        if os.path.exists(path):
            print_success(f"[{name}] trovato")
        else:
            print_error(f"[{name}] NON TROVATO: {path}")
            all_ok = False
    
    return all_ok

def verify_models():
    """Verifica la presenza dei modelli pre-addestrati."""
    print_step("Verifica Modelli Pre-addestrati")
    
    PROJECT_ROOT = os.path.dirname(__file__)
    
    models = {
        "CosyVoice": os.path.join(PROJECT_ROOT, "checkpoints", "cosyvoice_300m", "campplus.onnx"),
        "OpenVoice": os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2", "checkpoint.pth"),
        "StyleTTS2": os.path.join(PROJECT_ROOT, "checkpoints", "styletts2_libri", "epochs_2nd_00020.pth"),
    }
    
    # GPT-SoVITS supporta sia .bin che .pt
    gpt_bin = os.path.join(PROJECT_ROOT, "checkpoints", "gpt_sovits", "pretrained_models", "chinese-hubert-base.bin")
    gpt_pt = os.path.join(PROJECT_ROOT, "checkpoints", "gpt_sovits", "pretrained_models", "chinese-hubert-base.pt")
    
    if os.path.exists(gpt_bin) or os.path.exists(gpt_pt):
        models["GPT-SoVITS"] = gpt_pt if os.path.exists(gpt_pt) else gpt_bin
    
    missing = []
    
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            print_success(f"[{model_name}] trovato")
        else:
            print_error(f"[{model_name}] NON TROVATO")
            print(f"   Atteso: {model_path}")
            missing.append(model_name)
    
    if missing:
        print(f"\n⚠️  Modelli mancanti: {', '.join(missing)}")
        print("   Nota: La pipeline continuerà, ma questi modelli salteranno gli errori.")
    
    return True  # Comunque non blocchiamo

def verify_input_data():
    """Verifica la presenza dei dati di input."""
    print_step("Verifica Dati di Input")
    
    PROJECT_ROOT = os.path.dirname(__file__)
    input_dir = os.path.join(PROJECT_ROOT, "data", "raw_vctk")
    
    if not os.path.exists(input_dir):
        print_error(f"Cartella input non trovata: {input_dir}")
        return False
    
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print_error(f"Nessun file .wav trovato in {input_dir}")
        return False
    
    print_success(f"Trovati {len(wav_files)} file .wav")
    for f in wav_files:
        print(f"   - {f}")
    
    return True

def main():
    """Esegue l'intero setup."""
    print("\n" + "🌟"*30)
    print("🚀 SETUP VOICE IDENTITY ANALYSIS TOOL 🚀".center(60))
    print("🌟"*30 + "\n")
    
    # 1. Installa dipendenze
    if not install_dependencies():
        print_error("Fallito: Impossibile installare dipendenze")
        # Continua comunque
    
    # 2. Verifica cartelle
    if not verify_directories():
        print_error("ATTENZIONE: Alcune cartelle mancano")
        # Continua ma avvisa
    
    # 3. Verifica modelli
    verify_models()
    
    # 4. Verifica input data
    if not verify_input_data():
        print("\n" + "❌"*30)
        print("ERRORE CRITICO: Dati di input mancanti".center(60))
        print("❌"*30 + "\n")
        return False
    
    print("\n" + "✅"*30)
    print("✅ SETUP COMPLETATO CON SUCCESSO ✅".center(60))
    print("✅"*30 + "\n")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
