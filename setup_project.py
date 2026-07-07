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

def _pip_install(args, description):
    """Esegue 'pip install <args>' e ritorna True solo se il comando ha successo."""
    print(f"📦 {description}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *args],
        check=False
    )
    return result.returncode == 0

def _module_importable(module_name):
    """Verifica che un modulo sia realmente importabile in QUESTO interprete."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=False,
        capture_output=True
    )
    return result.returncode == 0

def install_dependencies():
    """Installa i pacchetti da requirements e le librerie esterne necessarie.

    A differenza della versione precedente, questa funzione NON stampa successo
    in modo incondizionato: controlla il return code di ogni installazione e
    verifica che 'openvoice' sia davvero importabile prima di ritornare True.
    """
    print_step("Installazione Dipendenze Python")

    req_file = os.path.join(os.path.dirname(__file__), "requirements_full.txt")
    if not os.path.exists(req_file):
        req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
        if not os.path.exists(req_file):
            return print_error("Nessun file requirements trovato.")

    try:
        # 1. Requirements standard.
        #    Non bloccante: alcuni pacchetti nel file possono essere opzionali.
        _pip_install(["-r", req_file], f"Installazione da {os.path.basename(req_file)}...")

        # 2. HuggingFace Hub: serve per il download dei modelli, quindi è bloccante.
        if not _module_importable("huggingface_hub"):
            if not _pip_install(["huggingface_hub"], "Installazione HuggingFace Hub..."):
                return print_error("Installazione di huggingface_hub fallita.")

        # 3. PyAV: serve solo assicurarsi che 'av' sia già presente nell'ambiente
        #    (una versione moderna, es. 17.x). NON lasciamo che OpenVoice lo
        #    reinstalli: il suo setup pinna faster-whisper==0.9.0 -> av==10.*, che
        #    non compila con Cython 3 (errore su av/logging.pyx).
        if not _module_importable("av"):
            if not _pip_install(["av"], "Installazione PyAV (wheel precompilata)..."):
                return print_error(
                    "Impossibile installare 'av'. Verifica requirements_full.txt."
                )

        # 4. OpenVoice da GitHub, MA senza le sue dipendenze.
        #    --no-deps installa solo il codice Python di 'openvoice': le librerie
        #    che gli servono a runtime (torch, librosa, soundfile, faster-whisper,
        #    av, ...) sono già nell'ambiente in versioni moderne e funzionanti.
        #    Senza --no-deps, pip tenta di compilare le versioni pinnate da
        #    OpenVoice (numpy==1.22.0, av==10.*) che non buildano su Python/
        #    Cython moderni, e l'installazione fallisce.
        if not _pip_install(
            ["--no-deps", "git+https://github.com/myshell-ai/OpenVoice.git"],
            "Installazione OpenVoice (da GitHub, senza dipendenze)..."
        ):
            print_error("Installazione di OpenVoice fallita.")
            print("   Riprova a mano: pip install --no-deps "
                  "\"git+https://github.com/myshell-ai/OpenVoice.git\"")
            return False

        # 5. Verifica REALE: il modulo deve essere importabile, non basta che pip
        #    non abbia dato errore.
        if not _module_importable("openvoice"):
            return print_error(
                "'openvoice' non è importabile dopo l'installazione. "
                "Controlla l'output di pip qui sopra."
            )

        print_success("Dipendenze installate e 'openvoice' importabile")
        return True

    except subprocess.TimeoutExpired:
        return print_error("Timeout durante l'installazione")
    except Exception as e:
        return print_error(f"Errore durante pip install: {e}")

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
    
    # 1. Installa dipendenze.
    #    Bloccante: senza 'openvoice' la pipeline crasha subito dopo, quindi
    #    ci fermiamo qui con un messaggio chiaro invece di proseguire.
    if not install_dependencies():
        print_error("Setup interrotto: dipendenze non installate correttamente.")
        print("   Risolvi l'installazione (vedi messaggi sopra) e rilancia.")
        return False
    
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