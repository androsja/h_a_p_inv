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

import logging
import os
import threading
from pathlib import Path

import joblib
import numpy as np

from shared import config

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_PATH          = config.DATA_DIR / "neural_model.joblib"
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD 
MIN_SAMPLES         = 8       # Mínimo de trades antes de confiar en el MLP

# Encoding de regímenes (debe coincidir con detect_regime en indicators.py)
REGIME_ENCODING = {
    "TREND_UP":   0,
    "TREND_DOWN": 1,
    "RANGE":      2,
    "NEUTRAL":    3,
    "MOMENTUM":   4,
    "CHAOS":      5,
}


# ── Clase Principal ───────────────────────────────────────────────────────────
class NeuralTradeFilter:
    """
    Filtro neuronal online. Aprende con cada trade completado.

    Features de entrada (8 valores normalizados):
        [rsi, macd_hist, atr_pct, vol_ratio, ema_spread_pct,
         zscore_vwap, regime_encoded, num_confirmations]
    """

    def __init__(self):
        self._model = None
        self._X: list[list[float]] = []
        self._y: list[int]  = []   # 1 = trade ganador, 0 = perdedor
        self._lock = threading.Lock()
        self._load()

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _load(self):
        try:
            if MODEL_PATH.exists():
                bundle = joblib.load(MODEL_PATH)
                self._model = bundle.get("model")
                self._X     = bundle.get("X", [])
                self._y     = bundle.get("y", [])
                log.info(
                    f"🧠 Red Neuronal cargada — {len(self._y)} trades en memoria, "
                    f"modelo={'activo' if self._model else 'cold-start'}"
                )
        except Exception as e:
            log.warning(f"No se pudo cargar el modelo neural: {e}")

    def _save(self):
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": self._model, "X": self._X, "y": self._y}, MODEL_PATH)
        except Exception as e:
            log.warning(f"No se pudo guardar el modelo neural: {e}")

    # ── API Pública ───────────────────────────────────────────────────────────

    def build_features(
        self,
        rsi: float,
        macd_hist: float,
        atr_pct: float,
        vol_ratio: float,
        ema_fast: float,
        ema_slow: float,
        zscore_vwap: float,
        regime: str,
        num_confirmations: int,
    ) -> list[float]:
        """Construye el vector de features normalizado para el modelo."""
        ema_spread_pct = ((ema_fast - ema_slow) / ema_slow * 100) if ema_slow else 0.0
        regime_enc = REGIME_ENCODING.get(regime, 3)  # Default NEUTRAL=3

        return [
            rsi / 100.0,                    # 0–1
            np.tanh(macd_hist),             # -1–1
            min(atr_pct / 5.0, 1.0),       # 0–1 (cap 5%)
            min(vol_ratio / 5.0, 1.0),     # 0–1 (cap 5x)
            np.tanh(ema_spread_pct),        # -1–1
            np.clip(zscore_vwap / 3.0, -1, 1),  # -1–1
            regime_enc / 5.0,              # 0–1
            min(num_confirmations / 7.0, 1.0),  # 0–1
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
        """
        from sklearn.neural_network import MLPClassifier

        with self._lock:
            self._X.append(features)
            self._y.append(1 if won else 0)

            n = len(self._y)
            log.info(f"🧠 Red Neuronal — nuevo trade registrado ({'WIN' if won else 'LOSS'}). Total: {n} muestras.")

            if n >= MIN_SAMPLES:
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
                "mode":            "MLP" if (n >= MIN_SAMPLES and self._model) else "cold-start",
                "threshold":       CONFIDENCE_THRESHOLD,
            }

    # ── Heurísticas Cold-Start ────────────────────────────────────────────────

    def _heuristic_predict(self, features: list[float], n: int) -> tuple[float, str]:
        """
        Reglas derivadas del análisis estadístico de 14 sesiones cuando
        aún no hay suficientes datos para el MLP.
        """
        rsi_norm        = features[0]   # 0-1
        macd_norm       = features[1]   # -1 a 1
        atr_norm        = features[2]   # 0-1
        vol_norm        = features[3]   # 0-1
        regime_enc_norm = features[6]   # 0-1
        num_conf_norm   = features[7]   # 0-1

        rsi         = rsi_norm * 100
        regime_enc  = round(regime_enc_norm * 5)  # 0-5
        num_conf    = round(num_conf_norm * 7)

        score = 0.5  # Inicio neutro
        reasons = []

        # RSI > 72 en TREND_UP (antes 68) → penalizar si es extremo
        if regime_enc == 0 and rsi > 72:   # TREND_UP muy tardío
            score -= 0.25
            reasons.append(f"TREND_UP muy tardío RSI={rsi:.0f}")

        # RANGE con MACD sin fuerza → falsas señales
        if regime_enc == 2 and abs(macd_norm) < 0.1:
            score -= 0.15
            reasons.append("RANGE: MACD débil")

        # Buenas confirmaciones → subir score
        if num_conf >= 4:
            score += 0.15
            reasons.append(f"{num_conf} confirmaciones")

        # Volumen alto → impulso real
        if vol_norm > 0.6:
            score += 0.10
            reasons.append("volumen fuerte")

        # ATR muy alto → volatilidad destructiva
        if atr_norm > 0.7:
            score -= 0.10
            reasons.append("ATR muy alto")

        # TREND_DOWN histórico es rentable → beneficiar
        if regime_enc == 1:
            score += 0.10
            reasons.append("TREND_DOWN favorable")

        score = float(np.clip(score, 0.0, 1.0))
        reason_str = f"Heurístico ({n}/{MIN_SAMPLES} muestras): {', '.join(reasons) or 'normal'}. P={score:.0%}"
        return score, reason_str


# ── Singleton global ──────────────────────────────────────────────────────────
_filter_instance: NeuralTradeFilter | None = None
_filter_lock = threading.Lock()


def get_neural_filter() -> NeuralTradeFilter:
    """Devuelve la instancia singleton del filtro neural."""
    global _filter_instance
    if _filter_instance is None:
        with _filter_lock:
            if _filter_instance is None:
                _filter_instance = NeuralTradeFilter()
    return _filter_instance
