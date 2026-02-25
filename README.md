# Project Layout

```
multimediaProject
┣ checkpoints                            # some additional data for the models
┣ data
┃ ┣ embeddings                           # embeddings of the samples from raw_vctk, the embeddings are suddivied by the encoder type
┃ ┗ raw_vctk                             # our samples, both real and fake ones
┣ model_src                              # models of voice cloning machines
┃ ┣ Open Voice
┃ ┣ StyleTTS2
┃ ┣ GPT-SoVITS
┃ ┗ CosyVoice
┣ src                                    # our code
┃ ┣ __pycache__                          # I don't... know
┃ ┣ extractors                           # extractors for every voice cloning machine      
┃ ┃ ┣ cosyvoice_wrapper.py               # extractor for cosyvoice
┃ ┃ ┣ gpt_sovits_wrapper.py              # extractor for GPT-SoVITS
┃ ┃ ┣ openvoice_wrapper.py               # extractor for openvoice
┃ ┃ ┗ styletts2_wrapper.py               # extractor for styletts2
┃ ┗ super_wrapper.py                     # it calls all the extractors for a single data
┣ total_wrapper.py                       # it calls the super_wrapper on all the data
┣ total_embed.py                         # it does all the embeddings and it does the graphs
┣ main.py                                # main code, it does run everything
┣ environment.yml                        # environment's settings 
┣ README.md                              # readme
┗ requirements.txt                       # libraries needed
```

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

**Comando per fare l'upgrade dell'ambiente conda esistente in seguito all'upgrade di nuove dipendenze**
conda env update --file environment.yml --prune

## Comandi venv per ambiente virtuale (se non si vuole usare Conda)
python -m venv venv
# Su Windows:
.\venv\Scripts\activate
# Su Linux/Mac:
source venv/bin/activate
