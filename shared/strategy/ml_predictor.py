import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime
from shared import config
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from shared.utils.logger import log
import joblib
from pathlib import Path

DATASET_PATH = config.ML_DATASET_FILE
MODEL_PATH   = config.DATA_DIR / "models" / "alpha_optimizer_model.joblib"

class MLPredictor:
    def __init__(self):
        self.classifier = None # Predice Win/Loss
        self.regressor = None  # Predice PnL esperado ($)
        self.is_trained = False
        self.accuracy = 0.0
        self.mae = 0.0 # Mean Absolute Error del regresor
        self.min_samples = 30  
        self.blocking_count = 0 
        self.alpha_blocking_count = 0 # Trades bloqueados por bajo valor esperado
        self._frozen: bool = False   
        self._lock = threading.Lock()
        
        # Mapeo de regímenes a números para la IA
        self.regime_map = {
            "TREND_UP": 2, "MOMENTUM": 3, "NEUTRAL": 1, 
            "RANGE": 0, "TREND_DOWN": -1, "CHAOS": -2
        }
        
        # 📂 Asegurar que la carpeta de modelos existe
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_and_train()

    def _load_and_train(self):
        # 1. Intentar cargar modelo ya entrenado desde disco
        # Solo lo cargamos si el aprendizaje está CONGELADO. 
        # Si estamos en modo entrenamiento, ignoramos el disco para forzar el aprendizaje de nuevos datos.
        freeze_learning = os.getenv("FREEZE_LEARNING", "false").lower() == "true"
        
        if MODEL_PATH.exists() and freeze_learning:
            try:
                with self._lock:
                    loaded_data = joblib.load(MODEL_PATH)
                    self.classifier = loaded_data['classifier']
                    self.regressor = loaded_data['regressor']
                    self.accuracy = loaded_data.get('accuracy', 0.0)
                    self.mae = loaded_data.get('mae', 0.0)
                    self.is_trained = True
                log.info(f"🧠 [AlphaOptimizer] Cargado (Modo Consulta). Precisión: {self.accuracy*100:.1f}%")
                return
            except Exception as e:
                log.warning(f"⚠️ [MLPredictor] No se pudo cargar: {e}")
        
        if not freeze_learning:
            log.info("🔥 [AlphaOptimizer] MODO APRENDIZAJE ACTIVO: Ignorando cerebro viejo para evolucionar.")

        # 2. Si no hay modelo o falló, entrenar desde CSV
        if not DATASET_PATH.exists():
            return
        
        with self._lock:
            try:
                # Leer dataset
                df = pd.read_csv(DATASET_PATH)
                log.info(f"📊 [MLPredictor] Procesando dataset para entrenamiento ({len(df)} filas)...")
                
                # Variables predictoras (features) evolucionadas
                features = ['rsi', 'macd_hist', 'ema_diff_pct', 'vwap_dist_pct', 'atr_pct', 'regime_code']
                target_cls = 'is_win'
                target_reg = 'pnl'
                
                # Manejar datasets antiguos que no tengan regime_code
                if 'regime_code' not in df.columns:
                    df['regime_code'] = 1 # Neutral por defecto
                
                # Verificar presencia de columnas críticas
                cols_to_check = ['rsi', 'macd_hist', 'ema_diff_pct', 'vwap_dist_pct', 'atr_pct', target_cls, target_reg]
                missing = [c for c in cols_to_check if c not in df.columns]
                if missing:
                    log.error(f"❌ [AlphaOptimizer] Faltan columnas críticas: {missing}")
                    return

                df = df.dropna(subset=cols_to_check)
                
                if len(df) < self.min_samples:
                    log.warning(f"⚠️ [AlphaOptimizer] Datos insuficientes ({len(df)}). Min: {self.min_samples}")
                    return

                X = df[features]
                y_cls = df[target_cls].astype(int)
                y_reg = df[target_reg].astype(float)

                log.info(f"⚙️ [AlphaOptimizer] Entrenando Doble Capa con {len(df)} muestras...")
                
                # 1. Entrenar Clasificador (Ofensivo)
                self.classifier = RandomForestClassifier(
                    n_estimators=150, 
                    max_depth=8, 
                    random_state=42, 
                    class_weight='balanced'
                )
                self.classifier.fit(X, y_cls)
                
                # 2. Entrenar Regresor (Home Runs)
                self.regressor = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
                self.regressor.fit(X, y_reg)

                self.is_trained = True
                self.accuracy = float(self.classifier.score(X, y_cls))
                
                # Calcular MAE simple
                preds_reg = self.regressor.predict(X)
                self.mae = float(np.mean(np.abs(preds_reg - y_reg)))
                
                # 💾 Guardar modelos
                joblib.dump({
                    'classifier': self.classifier,
                    'regressor': self.regressor,
                    'accuracy': self.accuracy,
                    'mae': self.mae,
                    'trained_at': datetime.now().isoformat(),
                    'samples_count': len(df)
                }, MODEL_PATH)
                
                log.info(f"✅ [AlphaOptimizer] Sistema de Doble Capa GUARDADO.")
                
            except Exception as e:
                log.error(f"❌ [MLPredictor] Error crítico en entrenamiento: {e}")
                import traceback
                log.error(traceback.format_exc())

    def predict_win(self, features: dict, regime: str = "NEUTRAL") -> tuple[bool, float, float]:
        """
        Sistema Alpha-Optimizer:
        1. Clasifica Probabilidad (Win/Loss)
        2. Estima Valor (PnL Esperado)
        Retorna (Aprobado, Probabilidad, PnL_Esperado)
        """
        if not self.is_trained or self.classifier is None:
            return True, 0.5, 0.0
            
        try:
            # Enriquecer features con régimen
            features_copy = features.copy()
            features_copy['regime_code'] = self.regime_map.get(regime, 1)
            
            feature_names = ['rsi', 'macd_hist', 'ema_diff_pct', 'vwap_dist_pct', 'atr_pct', 'regime_code']
            x_input = pd.DataFrame([features_copy], columns=feature_names)
            
            # Capa 1: Probabilidad
            prob_win = self.classifier.predict_proba(x_input)[0][1]
            pred_win = bool(prob_win > 0.55) # Umbral de confianza subido
            
            # Capa 2: Valor Esperado (PnL)
            expected_pnl = float(self.regressor.predict(x_input)[0])
            
            # FILTRO ALPHA (Maximización de Ganancias)
            # Solo aprobamos si la IA está segura Y el premio vale la pena (> $1.50 estimado)
            is_alpha_trade = pred_win and (expected_pnl > 1.5)
            
            if not pred_win:
                self.blocking_count += 1
            elif not is_alpha_trade:
                self.alpha_blocking_count += 1
                log.info(f"🚫 [AlphaOptimizer] Trade bloqueado por BAJO VALOR: Prob {prob_win:.1%} | PnL Est: ${expected_pnl:.2f}")
            
            return is_alpha_trade, float(prob_win), expected_pnl
            
        except Exception as e:
            log.error(f"Error en AlphaOptimizer: {e}")
            return True, 0.5, 0.0

    def freeze(self) -> None:
        """❄️ Congela el modelo: no guarda nuevos trades ni reentrena el Random Forest."""
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

    def save_trade(self, symbol: str, features: dict, pnl: float, regime: str = "NEUTRAL"):
        """Guarda resultados enriquecidos."""
        if self._frozen: return
        
        try:
            is_win = 1 if pnl > 0 else 0
            regime_code = self.regime_map.get(regime, 1)
            
            row = {
                'symbol': symbol,
                'pnl': pnl,
                'is_win': is_win,
                'regime_code': regime_code,
                **features
            }
            
            df_new = pd.DataFrame([row])
            
            should_retrain = False
            with self._lock:
                if DATASET_PATH.exists():
                    df_new.to_csv(DATASET_PATH, mode='a', header=False, index=False)
                    # Re-entrenar cada 10 trades para no sobrecargar CPU pero mantener aprendizaje
                    # (Aproximación simple: si el dataset tiene longitud múltiplo de 10)
                    try:
                        # No es la forma más eficiente (leer todo para contar), pero es segura 
                        # para este volumen de datos (160 empresas ~ 1000-2000 trades)
                        count = sum(1 for line in open(DATASET_PATH)) - 1 # -1 por el header
                        if count > self.min_samples and count % 10 == 0:
                            should_retrain = True
                    except: pass
                else:
                    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                    df_new.to_csv(DATASET_PATH, mode='w', header=True, index=False)
            
            if should_retrain:
                log.info(f"🔄 [MLPredictor] Re-entrenando modelo Forest con {count} muestras...")
                self._load_and_train()
                
        except Exception as e:
            log.error(f"Error guardando trade para ML: {e}")

    def get_sample_count(self) -> int:
        """Retorna el número total de muestras en el dataset."""
        try:
            if DATASET_PATH.exists():
                return sum(1 for _ in open(DATASET_PATH, encoding="utf-8")) - 1
        except:
            pass
        return 0

# Instancia global (singleton para no recargar a cada momento)
ml_predictor = MLPredictor()
