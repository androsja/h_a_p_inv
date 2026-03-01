import pandas as pd
import numpy as np
import os
from shared import config
from sklearn.ensemble import RandomForestClassifier
from shared.utils.logger import log
import collections

DATASET_PATH = config.ML_DATASET_FILE

class MLPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.min_samples = 20  # Requiere al menos 20 trades en el historial para entrenar
        self.blocking_count = 0 # Contador de trades bloqueados por ML en la sesión actual
        self._frozen: bool = False   # ❄️ Si True, no guarda nuevos trades ni reentrena
        self._load_and_train()

    def _load_and_train(self):
        if not DATASET_PATH.exists():
            return
        
        try:
            df = pd.read_csv(DATASET_PATH)
            if len(df) < self.min_samples:
                return  # No hay suficientes datos para un modelo estadísticamente relevante
                
            # Solo queremos trades cerrados. Asumimos que todos en el CSV están cerrados.
            # Variables predictoras (features):
            features = ['rsi', 'macd_hist', 'ema_diff_pct', 'vwap_dist_pct', 'atr_pct']
            target = 'is_win'
            
            # Limpiar datos nulos
            df = df.dropna(subset=features + [target])
            if len(df) < self.min_samples:
                return

            X = df[features]
            y = df[target]

            # Entrenar un Random Forest Classifier (robusto a no-linealidades)
            self.model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=5, 
                random_state=42, 
                class_weight='balanced'
            )
            self.model.fit(X, y)
            self.is_trained = True
            
            # Evaluar precisión en su propio set de entrenamiento (básico)
            acc = self.model.score(X, y)
            log.info(f"🧠 Modelo ML entrenado con {len(df)} ejemplos (Precisión interna: {acc*100:.1f}%)")
            
        except Exception as e:
            log.error(f"Error entrenando modelo ML: {e}")

    def predict_win(self, features: dict) -> tuple[bool, float]:
        """
        Predice si un trade será ganador (True) o perdedor (False) en base a la memoria.
        Retorna (predicción, probabilidad_de_ganar).
        Si el modelo no está entrenado, asume True (operar normalmente).
        """
        if not self.is_trained or self.model is None:
            return True, 0.5
            
        try:
            # Construir vector de entrada en el orden correcto
            feature_names = ['rsi', 'macd_hist', 'ema_diff_pct', 'vwap_dist_pct', 'atr_pct']
            x_input = pd.DataFrame([features], columns=feature_names)
            
            # vector[0] es la clase predominante predicha, predict_proba da las probabilidaes [prob_0, prob_1]
            pred_class = self.model.predict(x_input)[0]
            prob_win = self.model.predict_proba(x_input)[0][1]
            
            is_win = bool(pred_class == 1)
            if not is_win:
                self.blocking_count += 1
            
            return is_win, float(prob_win)
        except Exception as e:
            log.error(f"Error prediciendo trade con ML: {e}")
            return True, 0.5  # Si hay error, opera normal

    def freeze(self) -> None:
        """❄️ Congela el modelo: no guarda nuevos trades ni re-entrena el Random Forest."""
        self._frozen = True
        from shared.utils.logger import log as _l
        _l.info("🧊 [MLPredictor] Modelo RF CONGELADO. No se guardarán nuevos trades en el dataset.")

    def unfreeze(self) -> None:
        """🔥 Descongela: vuelve a guardar trades y entrenar."""
        self._frozen = False
        from shared.utils.logger import log as _l
        _l.info("🔥 [MLPredictor] Modelo RF DESCONGELADO.")

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def save_trade(self, symbol: str, features: dict, pnl: float):
        """Guarda los resultados del trade en el recolector de memoria.
        Si el modelo está congelado, no hace nada."""
        # ❄️ Congelado: ignorar el guardado
        if self._frozen:
            from shared.utils.logger import log as _l
            _l.debug("🧊 [MLPredictor] save_trade() ignorado — modelo congelado.")
            return
        try:
            is_win = 1 if pnl > 0 else 0
            
            row = {
                'symbol': symbol,
                'pnl': pnl,
                'is_win': is_win,
                **features
            }
            
            df_new = pd.DataFrame([row])
            
            if DATASET_PATH.exists():
                df_new.to_csv(DATASET_PATH, mode='a', header=False, index=False)
            else:
                DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                df_new.to_csv(DATASET_PATH, mode='w', header=True, index=False)
                
            # Opcional: Reentrenar el modelo dinámicamente cada vez que se guardan 10 nuevos trades (por optimización)
            # Para este scope, se entrenará en el siguiente inicio de sesión.
                
        except Exception as e:
            log.error(f"Error guardando trade para ML: {e}")

# Instancia global (singleton para no recargar a cada momento)
ml_predictor = MLPredictor()
