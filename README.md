# Voice Identity Analysis: OpenVoice V2 vs StyleTTS2

Questo progetto confronta le capacità di estrazione dell'identità vocale (voice embedding) tra OpenVoice V2 e StyleTTS2 utilizzando il dataset VCTK.

## Requisiti Hardware
- **GPU**: NVIDIA (consigliata 4GB+ VRAM)
- **OS**: Windows (testato) / Linux

## Installazione Rapida

1. **Clona il repository e i sottomoduli:**
   ```bash
   git clone --recursive <link-tuo-repo>
   cd multimediaProject

## Comandi Conda per installare le dipendenze
conda env create -f environment.yml
conda activate openvoice_env

## Comandi venv per ambiente virtuale (se non si vuole usare Conda)
python -m venv venv
# Su Windows:
.\venv\Scripts\activate
# Su Linux/Mac:
source venv/bin/activate