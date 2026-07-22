#!/bin/bash
# Lancia la pipeline di estrazione in sequenza su train, val, test
# dell'ITASpoof dataset, senza intervento manuale tra uno split e l'altro.
#
# Uso: bash run_all_splits.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITASPOOF_ROOT="/media/SSD_new/ITASpoof/detection"
OUTPUT_ROOT="/media/SSD_new/projects/MDS2025_voice_cloning"

SPLITS=("val" "train" "test")

for split in "${SPLITS[@]}"; do
    echo ""
    echo "=================================================="
    echo "AVVIO SPLIT: $split"
    echo "=================================================="

    export VOICE_INPUT_DIR="$ITASPOOF_ROOT/$split"
    export VOICE_DATA_DIR="$OUTPUT_ROOT/output_$split"

    echo "Input:  $VOICE_INPUT_DIR"
    echo "Output: $VOICE_DATA_DIR"

    python3 "$PROJECT_ROOT/main.py"

    echo "Split '$split' completato."
done

echo ""
echo "=================================================="
echo "TUTTI GLI SPLIT COMPLETATI (val, train, test)"
echo "=================================================="
