"""
agent.py
========

Kerangka kerja untuk semua 'otak' atau 'agent' pembuat keputusan.
Setiap agent harus mewarisi dari BaseAgent dan mengimplementasikan
metode `decide()`.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    Kelas dasar abstrak untuk semua agent.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Metode utama untuk membuat keputusan trading.

        Args:
            opportunity (Dict[str, Any]): Sebuah dictionary yang berisi semua
                                         informasi tentang peluang trading
                                         (skor, komponen, fitur, dll.).

        Returns:
            str: Keputusan trading, contoh: "ACCEPT", "REJECT", "WAIT".
        """
        pass

    def __str__(self):
        return f"Agent(name={self.name})"


class RuleBasedAgent(BaseAgent):
    """
    Agent sederhana yang membuat keputusan berdasarkan aturan-aturan dasar.
    Contoh: Cek skor dan beberapa fitur kunci.
    """
    def __init__(self, confidence_threshold: float = 5.0):
        super().__init__(name="RuleBasedAgent")
        self.confidence_threshold = confidence_threshold
        print(f"INFO: {self.name} diinisialisasi dengan threshold: {self.confidence_threshold}")

    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Membuat keputusan berdasarkan skor dari sinyal.
        """
        score = opportunity.get('score', 0)

        if abs(score) >= self.confidence_threshold:
            decision = "ACCEPT"
            print(f"INFO: {self.name} memutuskan: {decision} (Skor {score:.2f} >= Threshold {self.confidence_threshold})")
        else:
            decision = "REJECT"
            print(f"INFO: {self.name} memutuskan: {decision} (Skor {score:.2f} < Threshold {self.confidence_threshold})")

        return decision


# --- Agent Cerdas dengan Neural Network ---
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

class NeuralAgent(BaseAgent):
    """
    Agent yang menggunakan model Jaringan Syaraf Tiruan (MLPClassifier)
    yang sudah dilatih untuk membuat keputusan.
    """
    def __init__(self, model_path: str):
        super().__init__(name="NeuralAgent")
        self.model = self._load_model(model_path)
        print(f"INFO: {self.name} diinisialisasi dengan model dari: {model_path}")

    def _load_model(self, model_path: str) -> MLPClassifier:
        """Memuat model yang sudah dilatih dari file."""
        try:
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            print(f"ERROR: File model tidak ditemukan di {model_path}. Agent tidak akan berfungsi.")
            return None
        except Exception as e:
            print(f"ERROR: Gagal memuat model dari {model_path}: {e}")
            return None

    def decide(self, opportunity: Dict[str, Any]) -> str:
        """
        Membuat keputusan menggunakan prediksi dari model neural network.
        """
        if self.model is None:
            print(f"WARNING: {self.name} tidak memiliki model, keputusan ditolak.")
            return "REJECT"

        features = opportunity.get('features')
        if features is None or features.size == 0:
            print(f"WARNING: {self.name} tidak menerima fitur untuk dianalisis, keputusan ditolak.")
            return "REJECT"

        # Model scikit-learn mengharapkan input 2D
        features_2d = np.array(features).reshape(1, -1)

        # Prediksi probabilitas (kelas 1 adalah 'WIN' atau 'ACCEPT')
        try:
            probability_of_win = self.model.predict_proba(features_2d)[0][1]

            # Contoh logika keputusan: terima jika probabilitas > 70%
            if probability_of_win > 0.70:
                decision = "ACCEPT"
            else:
                decision = "REJECT"

            print(f"INFO: {self.name} memprediksi probabilitas keberhasilan {probability_of_win:.2%}. Keputusan: {decision}")
            return decision

        except Exception as e:
            print(f"ERROR: {self.name} gagal membuat prediksi: {e}")
            return "REJECT"
