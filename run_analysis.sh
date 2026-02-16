#!/bin/bash

ENV_NAME="openvoice_env"

echo "📂 Posizione attuale: $(pwd)"
echo "🚀 Attivazione ambiente: $ENV_NAME"

# Carica il setup di conda per lo script shell
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "❌ Errore: Impossibile attivare l'ambiente $ENV_NAME"
    exit 1
fi

python3 main.py

if [ $? -eq 0 ]; then
    echo "✅ Pipeline completata!"
else
    echo "❌ Errore durante l'esecuzione."
fi