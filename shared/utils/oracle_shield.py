import joblib
import os
import logging
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "shared" / "data"
MODELS_DIR = DATA_DIR / "neural_models"
ORACLE_MODEL_PATH = MODELS_DIR / "oracle_sentinel.joblib"

log = logging.getLogger("OracleShield")

class OracleShield:
    """
    Segunda capa de validación (El Oracle).
    Especializado en detectar 'Zonas de Muerte' y proteger el capital.
    """
    
    def __init__(self, threshold: float = 0.51):
        self.threshold = threshold
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Intenta cargar el modelo del Oracle si existe."""
        try:
            if ORACLE_MODEL_PATH.exists():
                self.model = joblib.load(ORACLE_MODEL_PATH)
                log.info(f"🛡️ [Oracle] Sentinel cargado (Umbral: {self.threshold})")
            else:
                # Si no existe, al principio será un 'Observador' pasivo
                log.info("🛡️ [Oracle] Sentinel en modo OBSERVADOR (Sin modelo previo)")
        except Exception as e:
            log.warning(f"Error cargando Oracle: {e}")

    def predict_quality(self, features: list[float]) -> dict:
        """
        Evalúa la calidad de una señal.
        Devuelve: { approved: bool, confidence: float, reason: str }
        """
        if self.model is None:
            return {"approved": True, "confidence": 1.0, "reason": "Modo Observador"}
            
        try:
            X = np.array([features])
            prob_win = float(self.model.predict_proba(X)[0, 1])
            
            is_approved = prob_win >= self.threshold
            
            return {
                "approved": is_approved,
                "confidence": round(prob_win, 3),
                "reason": "Probabilidad de Ganancia insuficiente" if not is_approved else "Calidad aceptable"
            }
        except Exception as e:
            log.error(f"Error en predicción Oracle: {e}")
            return {"approved": True, "confidence": 0.5, "reason": "Error en Auditoría"}

    def update_stats(self, won: bool, predicted_approved: bool):
        """Registra el desempeño del Oracle para medir su fiabilidad."""
        # TODO: Implementar persistencia de fiabilidad (Oracle vs Realidad)
        pass

# Instancia global para ser usada por el Orquestador/Bots
oracle = OracleShield()
