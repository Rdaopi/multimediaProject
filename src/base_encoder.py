from abc import ABC, abstractmethod
import numpy as np

class BaseVoiceEncoder(ABC):
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """Carica i pesi del modello specifico."""
        pass

    @abstractmethod
    def preprocess_audio(self, audio_path):
        """Carica e processa l'audio (resampling, trimming) per il modello specifico."""
        pass

    @abstractmethod
    def get_embedding(self, audio_path) -> np.ndarray:
        """
        Input: Path del file audio.
        Output: Numpy array del vettore latente (embedding).
        """
        pass