"""
utils/neural_filter.py — Filtro de Pre-Aprobación con Red Neuronal (MLP).

Antes de ejecutar cualquier señal BUY, este módulo calcula la probabilidad
de que el trade sea rentable usando un Multi-Layer Perceptron (MLP) entrenado
con el historial real de trades del bot.

Flujo:
    1. analyze() genera señal BUY
    2. neural_filter.predict(features) devuelve P(win) ∈ [0, 1]
    3. Si P(win) < CONFIDENCE_THRESHOLD → señal bloqueada (HOLD)
    4. Después de cada sesión, neural_filter.fit(features, won) re-entrena el modelo

Cold-Start: cuando aún no hay datos suficientes (<MIN_SAMPLES) se usa un set
de reglas heurísticas derivadas del análisis estadístico de las 14 sesiones.
"""

from __future__ import annotations
import hashlib
import logging
import os
import threading
from pathlib import Path

import joblib
import numpy as np

from shared import config

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_DIR           = config.DATA_DIR / "neural_models"  # Un archivo por símbolo
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD 
MIN_SAMPLES         = config.COLD_START_MIN_SAMPLES

# Encoding de regímenes (debe coincidir con detect_regime en indicators.py)
REGIME_ENCODING = {
    "TREND_UP":   0,
    "TREND_DOWN": 1,
    "RANGE":      2,
    "NEUTRAL":    3,
    "MOMENTUM":   4,
    "CHAOS":      5,
}

def _model_path_for(symbol: str) -> Path:
    """Devuelve el path del modelo para un símbolo específico."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    return MODEL_DIR / f"neural_{safe}.joblib"


# ── Clase Principal ───────────────────────────────────────────────────────────
class NeuralTradeFilter:
    """
    Filtro neuronal online. Aprende con cada trade completado.

    Features de entrada (13 valores normalizados):
        [h_norm, vwap_dist_pct, rsi, macd_hist, atr_pct, vol_ratio,
         ema_spread_pct, zscore_vwap, regime_encoded, num_confirmations,
         adx, has_pattern, is_adx_rising]
    """

    def __init__(self, symbol: str = "GLOBAL"):
        self._symbol = symbol.upper()
        self._model = None
        self._X: list[list[float]] = []
        self._y: list[int]  = []   # 1 = trade ganador, 0 = perdedor
        self._lock = threading.Lock()
        self._frozen: bool = False   # ❄️ Si True, no aprende ni sobreescribe el modelo
        self._load()

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _load(self):
        path = _model_path_for(self._symbol)
        try:
            if path.exists():
                bundle = joblib.load(path)
                self._model = bundle.get("model")
                self._X     = bundle.get("X", [])
                self._y     = bundle.get("y", [])
                
                # Validación de arquitectura
                if self._X:
                    n_features = len(self._X[0])
                    if n_features != 13:
                        log.warning(f"⚠️ [{self._symbol}] Arquitectura cambió ({n_features}→13). Reseteando.")
                        self._model = None
                        self._X = []
                        self._y = []

                log.info(
                    f"🧠 [{self._symbol}] Modelo cargado — {len(self._y)} trades, "
                    f"estado={'activo' if self._model else 'cold-start'}"
                )
        except Exception as e:
            log.warning(f"No se pudo cargar modelo [{self._symbol}]: {e}")

    def _save(self):
        if self._frozen:
            log.debug(f"🧊 [{self._symbol}] Modelo congelado — guardado omitido.")
            return
        try:
            joblib.dump({"model": self._model, "X": self._X, "y": self._y}, _model_path_for(self._symbol))
        except Exception as e:
            log.warning(f"No se pudo guardar modelo [{self._symbol}]: {e}")

    # ── Control de Congelación ───────────────────────────────────────────────

    def freeze(self) -> None:
        """❄️ Congela el modelo: seguirá prediciendo pero NO aprenderá ni guardará cambios."""
        self._frozen = True
        log.info("🧊 [NeuralFilter] Modelo MLP CONGELADO. Solo lectura — nuevos trades no modificarán la IA.")

    def unfreeze(self) -> None:
        """🔥 Descongela: el modelo vuelve a aprender de nuevos trades."""
        self._frozen = False
        log.info("🔥 [NeuralFilter] Modelo MLP DESCONGELADO. Retomarán el aprendizaje.")

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # ── API Pública ───────────────────────────────────────────────────────────

    def build_features(
        self,
        symbol: str,
        hour_of_day: float,
        vwap_dist_pct: float,
        rsi: float,
        macd_hist: float,
        atr_pct: float,
        vol_ratio: float,
        ema_spread_pct: float,
        zscore_vwap: float,
        regime: str,
        num_confirmations: int,
        adx: float = 20.0,
        has_pattern: bool = False,
        is_adx_rising: bool = False,
    ) -> list[float]:
        """Construye el vector de features normalizado para el modelo."""
        regime_enc = REGIME_ENCODING.get(regime, 3)  # Default NEUTRAL=3
        
        # Normalización de Hora Militar Cíclica (0-23)
        h_norm = hour_of_day / 24.0

        return [
            h_norm,                             # 0-1 Normalizado de hora del día
            np.clip(vwap_dist_pct / 5.0, -1, 1),# -1–1 (cap al 5% distancia vwap)
            rsi / 100.0,                        # 0–1
            np.tanh(macd_hist),                 # -1–1
            min(atr_pct / 5.0, 1.0),           # 0–1 (cap 5%)
            min(vol_ratio / 5.0, 1.0),         # 0–1 (cap 5x)
            np.tanh(ema_spread_pct),            # -1–1
            np.clip(zscore_vwap / 3.0, -1, 1),  # -1–1
            regime_enc / 5.0,                  # 0–1
            min(num_confirmations / 7.0, 1.0),  # 0–1
            min(adx / 50.0, 1.0),              # 0–1 (cap 50 ADX)
            1.0 if has_pattern else 0.0,        # 0 o 1 (Documentación técnica probada)
            1.0 if is_adx_rising else 0.0,      # 0 o 1
        ]

    def predict(self, features: list[float]) -> tuple[float, str]:
        """
        Devuelve (probabilidad_de_win, razon).
        probabilidad ∈ [0.0, 1.0]. Razon es un texto explicativo.
        """
        with self._lock:
            n = len(self._y)

            # ── Modo Cold-Start (datos insuficientes) ────────────────────────
            if n < MIN_SAMPLES or self._model is None:
                return self._heuristic_predict(features, n)

            # ── Modo MLP ─────────────────────────────────────────────────────
            try:
                X = np.array([features])
                proba = self._model.predict_proba(X)[0][1]  # P(clase=1=win)
                reason = f"MLP ({n} trades entrenados): P(win)={proba:.0%}"
                return float(proba), reason
            except Exception as e:
                log.warning(f"Error en predicción MLP: {e}")
                return 0.5, "MLP error — usando neutro"

    def fit(self, features: list[float], won: bool) -> None:
        """
        Registra el resultado de un trade y re-entrena el modelo.
        Llamar después de cerrar cada posición.
        Si el modelo está congelado (frozen=True), esta función no hace nada.
        """
        from sklearn.neural_network import MLPClassifier

        # ❄️ Si está congelado, NO aprender nada nuevo
        if self._frozen:
            log.debug("🧊 [NeuralFilter] fit() ignorado — modelo congelado.")
            return

        with self._lock:
            self._X.append(features)
            self._y.append(1 if won else 0)

            n = len(self._y)
            log.info(f"🧠 Red Neuronal — nuevo trade registrado ({'WIN' if won else 'LOSS'}). Total: {n} muestras.")

            if n >= MIN_SAMPLES:
                # Evitar entrenar si no tenemos ambas clases (Wins y Losses)
                if len(set(self._y)) < 2:
                    log.info(f"🧠 MLP pospuesto: Aún no hay ambas clases (Wins/Losses) para entrenar ({n} muestras, todas clase {self._y[0]}).")
                    self._save()
                    return

                # Re-entrenar con warm_start para aprendizaje incremental
                try:
                    X_arr = np.array(self._X)
                    y_arr = np.array(self._y)

                    if self._model is None:
                        self._model = MLPClassifier(
                            hidden_layer_sizes=(32, 16, 8),
                            activation="relu",
                            solver="adam",
                            max_iter=500,
                            warm_start=True,
                            random_state=42,
                            alpha=0.01,   # L2 regularización — evita sobreajuste
                        )

                    self._model.fit(X_arr, y_arr)
                    acc = self._model.score(X_arr, y_arr)
                    from shared.utils.logger import log_training
                    log_training("NeuralFilter_MLP", n, float(acc), extra=f"last_won={won}")
                    log.info(f"🧠 MLP re-entrenado. Accuracy en train: {acc:.0%} ({n} muestras)")
                except Exception as e:
                    log.warning(f"Error entrenando MLP: {e}")

            self._save()

    def get_stats(self) -> dict:
        """Estadísticas del modelo para el dashboard."""
        with self._lock:
            n = len(self._y)
            wins = sum(self._y)
            acc = 0.0
            if n >= MIN_SAMPLES and self._model is not None:
                try:
                    X_arr = np.array(self._X)
                    y_arr = np.array(self._y)
                    acc = float(self._model.score(X_arr, y_arr))
                except Exception:
                    pass

            return {
                "total_samples":   n,
                "wins":            wins,
                "losses":          n - wins,
                "win_rate_hist":   round(wins / n * 100, 1) if n > 0 else 0.0,
                "model_accuracy":  round(acc * 100, 1),
                "ai_mode":         "MLP" if (n >= MIN_SAMPLES and self._model) else "cold-start",
                "threshold":       CONFIDENCE_THRESHOLD,
            }

    # ── Heurísticas Cold-Start ────────────────────────────────────────────────

    def _heuristic_predict(self, features: list[float], n: int) -> tuple[float, str]:
        """
        Reglas derivadas del análisis estadístico de 14 sesiones cuando
        aún no hay suficientes datos para el MLP.
        """
        rsi_norm        = features[2]   # 0-1
        macd_norm       = features[3]   # -1 a 1
        atr_norm        = features[4]   # 0-1
        vol_norm        = features[5]   # 0-1
        regime_enc_norm = features[8]   # 0-1
        num_conf_norm   = features[9]  # 0-1

        rsi         = rsi_norm * 100
        regime_enc  = round(regime_enc_norm * 5)  # 0-5
        num_conf    = round(num_conf_norm * 7)

        score = config.COLD_START_BASE_SCORE
        reasons = []

        # RSI > HEURISTIC_RSI_UPPER en TREND_UP → penalizar si es extremo
        if regime_enc == 0 and rsi > config.HEURISTIC_RSI_UPPER:
            score += config.HEURISTIC_PENALTY_RSI
            reasons.append(f"TREND_UP muy tardío RSI={rsi:.0f}")

        # RANGE con MACD sin fuerza → falsas señales
        if regime_enc == 2 and abs(macd_norm) < config.HEURISTIC_MACD_MIN:
            score += config.HEURISTIC_PENALTY_MACD
            reasons.append("RANGE: MACD débil")

        # Buenas confirmaciones → subir score sustancialmente
        if num_conf >= config.HEURISTIC_CONF_HIGH:
            score += config.HEURISTIC_BOOST_CONF_H
            reasons.append(f"{num_conf} confirmaciones")
        elif num_conf == config.HEURISTIC_CONF_MID:
            score += config.HEURISTIC_BOOST_CONF_M
            reasons.append(f"{num_conf} confirmaciones")

        # Volumen expansivo → impulso real
        if vol_norm > config.HEURISTIC_VOL_MIN:
            score += config.HEURISTIC_BOOST_VOL
            reasons.append("volumen expansivo")

        # ATR muy alto → volatilidad destructiva
        if atr_norm > config.HEURISTIC_ATR_MAX:
            score += config.HEURISTIC_PENALTY_ATR
            reasons.append("ATR destructor")

        # TREND_DOWN histórico es muy rentable → beneficiar agresivo
        if regime_enc == 1:
            score += config.HEURISTIC_BOOST_TREND
            reasons.append("TREND_DOWN (históricamente favorable)")

        score = float(np.clip(score, 0.0, 1.0))
        reason_str = f"Heurístico ({n}/{MIN_SAMPLES} muestras): {', '.join(reasons) or 'normal'}. P={score:.0%}"
        return score, reason_str


    def reset(self) -> None:
        """🧨 Borra TODO en memoria y en disco para este símbolo."""
        with self._lock:
            self._model  = None
            self._X      = []
            self._y      = []
            self._frozen = False
            try:
                p = _model_path_for(self._symbol)
                if p.exists():
                    p.unlink()
            except Exception as e:
                log.warning(f"No se pudo borrar modelo [{self._symbol}]: {e}")
            log.info(f"🧨 [{self._symbol}] RESET — modelo borrado de memoria y disco.")


# ── Registry por símbolo (reemplaza el singleton global) ─────────────────────
_filter_registry: dict[str, NeuralTradeFilter] = {}
_filter_lock = threading.Lock()


def get_neural_filter(symbol: str = "GLOBAL") -> NeuralTradeFilter:
    """Devuelve (o crea) la instancia del filtro neural para el símbolo dado.
    
    Cada símbolo tiene su propio modelo aislado para evitar interferencia
    entre activos con comportamientos de mercado distintos.
    """
    key = symbol.upper()
    if key not in _filter_registry:
        with _filter_lock:
            if key not in _filter_registry:
                _filter_registry[key] = NeuralTradeFilter(symbol=key)
    return _filter_registry[key]


def reset_neural_filter(symbol: str | None = None) -> None:
    """🧨 Resetea el modelo de un símbolo específico o de TODOS si symbol=None."""
    global _filter_registry
    with _filter_lock:
        if symbol:
            key = symbol.upper()
            if key in _filter_registry:
                _filter_registry[key].reset()
                del _filter_registry[key]
                log.info(f"🧨 [NeuralFilter] Modelo de {key} reseteado.")
        else:
            # WIPE TOTAL — borrar todos los modelos
            for inst in _filter_registry.values():
                inst.reset()
            _filter_registry.clear()
            # También borrar archivos viejos del formato anterior
            old = config.DATA_DIR / "neural_model.joblib"
            if old.exists():
                old.unlink()
            log.info("🧨 [NeuralFilter] RESET TOTAL — todos los modelos por símbolo borrados.")

