import os
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm  # Ti consiglio: pip install tqdm

class BaseVoiceEncoder(ABC):
    def __init__(self, device=None):
        self.device = device
        self.model = None  # Sarà inizializzato dalle sottoclassi
        self.load_model()  # Chiama il metodo astratto

    @abstractmethod
    def load_model(self):
        """Logica di caricamento del modello e dei pesi."""
        pass

    @abstractmethod
    def extract(self, audio_path) -> np.ndarray:
        """Logica specifica per ottenere l'embedding da un singolo file."""
        pass

    def process_all(self, input_dir, output_dir, suffix):
        """
        Metodo universale per processare una cartella.
        :param suffix: Il suffisso da dare al file salvato (es. '_ov.npy' o '_style.npy')
        """
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        
        if not files:
            print(f"⚠️ Nessun file .wav in {input_dir}")
            return

        print(f"🚀 Inizio estrazione ({self.__class__.__name__}): {len(files)} file...")
        successi = 0
        
        # tqdm crea una barra di progresso nel terminale
        for file in tqdm(files, desc="Processing", unit="file"):
            full_path = os.path.join(input_dir, file)
            emb = self.extract(full_path)
            
            if emb is not None:
                out_name = os.path.splitext(file)[0] + suffix
                np.save(os.path.join(output_dir, out_name), emb)
                successi += 1

        print(f"\n✅ Completato! {successi}/{len(files)} file salvati in: {output_dir}")