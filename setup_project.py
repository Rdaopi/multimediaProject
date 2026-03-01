"""
Setup script per il Voice Identity Analysis Tool.
Esegue tutte le verifiche preliminari e l'installazione delle dipendenze.
"""

import os
import sys
import subprocess
import shutil

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

def download_models():
    """Scarica i modelli e corregge automaticamente i percorsi dei file."""
    print_step("Download e Correzione Modelli")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=False)
        from huggingface_hub import hf_hub_download

    PROJECT_ROOT = os.path.dirname(__file__)
    
    models_to_fetch = [
        {
            "name": "CosyVoice",
            "repo": "FunAudioLLM/CosyVoice-300M",
            "file": "campplus.onnx",
            "dest": os.path.join(PROJECT_ROOT, "checkpoints", "cosyvoice_300m")
        },
        {
            "name": "OpenVoice V2 (Pesi)",
            "repo": "myshell-ai/OpenVoiceV2",
            "file": "converter/checkpoint.pth", # Il vero percorso su HuggingFace
            "dest": os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2")
        },
        {
            "name": "OpenVoice V2 (Config)",
            "repo": "myshell-ai/OpenVoiceV2",
            "file": "converter/config.json", # Il file di configurazione fondamentale
            "dest": os.path.join(PROJECT_ROOT, "checkpoints", "openvoice_v2")
        },
        {
            "name": "StyleTTS2",
            "repo": "yl4579/StyleTTS2-LibriTTS",
            "file": "Models/LibriTTS/epochs_2nd_00020.pth",
            "dest": os.path.join(PROJECT_ROOT, "checkpoints", "styletts2_libri")
        },
        {
            "name": "GPT-SoVITS (Config)",
            "repo": "TencentGameMate/chinese-hubert-base",
            "file": "config.json",
            "dest": os.path.join(PROJECT_ROOT, "checkpoints", "gpt_sovits", "pretrained_models")
        },
        {
            "name": "GPT-SoVITS (Pesi)",
            "repo": "TencentGameMate/chinese-hubert-base",
            "file": "pytorch_model.bin",
            "rename": "chinese-hubert-base.bin", # Magia: lo rinominiamo automaticamente per il codice
            "dest": os.path.join(PROJECT_ROOT, "checkpoints", "gpt_sovits", "pretrained_models")
        }
    ]

    for m in models_to_fetch:
        # Usa 'rename' se specificato, altrimenti usa il nome originale
        target_name = m.get("rename", os.path.basename(m["file"]))
        final_file_path = os.path.join(m["dest"], target_name)
        
        if not os.path.exists(final_file_path):
            print(f"⬇️ Scaricamento [{m['name']}]...")
            try:
                # Scarica il file (hf_hub_download mantiene la struttura delle cartelle del repo)
                downloaded_path = hf_hub_download(repo_id=m["repo"], filename=m["file"], local_dir=m["dest"])
                
                # FIX: Se il file è finito in una sottocartella (o deve essere rinominato), lo spostiamo nel percorso finale corretto
                if downloaded_path != final_file_path:
                    os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
                    shutil.move(downloaded_path, final_file_path)
                    
                    # Pulizia cartelle vuote create dal download
                    parent_dir = os.path.dirname(downloaded_path)
                    if parent_dir.startswith(m["dest"]) and parent_dir != m["dest"]:
                        shutil.rmtree(os.path.join(m["dest"], m["file"].split('/')[0]), ignore_errors=True)
                
                print_success(f"{m['name']} pronto.")
            except Exception as e:
                print(f"❌ Errore critico download {m['name']}: {e}")
        else:
            print_success(f"[{m['name']}] configurato correttamente.")

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
    
    # FIX: Registriamo sempre GPT-SoVITS per il controllo, così la funzione se ne accorge!
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
    
    # 3. Installa modelli
    download_models()

    # 4. Verifica modelli
    verify_models()
    
    # 5. Verifica input data
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