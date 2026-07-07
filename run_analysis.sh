#!/bin/bash

ENV_NAME="openvoice_env"

# =====================================================================
# CONFIGURAZIONE PERCORSI — l'unico punto da toccare per cambiare dove
# la pipeline legge l'input e scrive gli output.
#
#   VOICE_INPUT_DIR : cartella dei file .wav di input
#   VOICE_DATA_DIR  : cartella base per gli output (embeddings, db, grafici)
#
# Le variabili vengono ereditate da tutti gli script Python della pipeline
# (total_wrapper -> super_wrapper -> statistical_analysis).
#
# --- LOCALE (default): decommenta questi ---
export VOICE_INPUT_DIR="$(pwd)/data/raw_vctk"
export VOICE_DATA_DIR="$(pwd)/data"
#
# --- MACCHINA SUPERVISORE (192.168.163.9): commenta i due sopra e usa questi ---
# export VOICE_INPUT_DIR="/media/SSD_new/ITASpoof/detection"
# export VOICE_DATA_DIR="/media/SSD_new/projects/MDS2025_voice_cloning"
# =====================================================================

echo "📂 Posizione attuale: $(pwd)"
echo "📥 Input:  $VOICE_INPUT_DIR"
echo "📤 Output: $VOICE_DATA_DIR"

# Carica conda per questo script (necessario perché 'conda activate' funzioni
# dentro uno script non interattivo).
source "$(conda info --base)/etc/profile.d/conda.sh"

# Crea openvoice_env solo se non esiste già. Se c'è, viene lasciato com'è.
if ! conda env list | grep -qE "/envs/$ENV_NAME$"; then
    echo "🚀 Creazione ambiente Conda: $ENV_NAME..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
    conda create -n "$ENV_NAME" python=3.10 pip git -y
fi

echo "🚀 Attivazione ambiente: $ENV_NAME"
conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Errore: impossibile attivare l'ambiente $ENV_NAME"
    exit 1
fi

echo "🐍 Interprete in uso: $(which python3)"

python3 main.py

if [ $? -eq 0 ]; then
    echo "✅ Pipeline completata!"
else
    echo "❌ Errore durante l'esecuzione."
fi