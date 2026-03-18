"""
strategy/indicators.py ─ Motor de análisis técnico y toma de decisiones.

Orquesta el análisis multi-indicador llamando a los módulos especializados:
- indicators_calc.py (matemáticas)
- patterns.py        (velas)
- regimes.py         (mercado)
- signal_models.py   (objetos)
"""

import os
import joblib
import pandas as pd
from functools import lru_cache

from shared import config
from shared.utils.logger import log

# ── Importar módulos especializados ───────────────────────────────────────────
from shared.strategy.indicators_calc import (
    ema, rsi, adx, macd, atr, vwap, zscore_vwap, bollinger_bands, kama, super_smoother
)
from shared.strategy.patterns import volume_spike, detect_hammer, detect_engulfing
from shared.strategy.regimes  import detect_regime, REGIMEN_LABELS
from shared.strategy.signal_models import SignalResult

# Constantes de señal
SIGNAL_BUY  = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

MIN_BARS = 80

@lru_cache(maxsize=1)
def _load_ai_model(mtime):
    try:
        return joblib.load(config.AI_MODEL_FILE)
    except Exception:
        return None

def get_ai_model():
    p = config.AI_MODEL_FILE
    if os.path.exists(p):
        return _load_ai_model(os.path.getmtime(p))
    return None

def analyze(df: pd.DataFrame, symbol: str = "", asset_type: str = "normal") -> SignalResult:
    sym_prefix = f"[{symbol}] " if symbol else ""
    
    if len(df) < MIN_BARS:
        if symbol:
            log.warning(f"{sym_prefix}⏳ Datos insuficientes: {len(df)}/{MIN_BARS} barras.")
        return SignalResult(
            signal=SIGNAL_HOLD,
            ema_fast=0.0, ema_slow=0.0, ema_200=0.0, rsi_value=50.0, macd_hist=0.0,
            vwap_value=0.0, atr_value=0.0,
            close=float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0,
            timestamp=df.index[-1] if len(df) > 0 else None,
            blocks=[f"⏳ Cargando datos ({len(df)}/{MIN_BARS})"],
        )

    prices = df["Close"].astype(float)
    
    # 1. Calcular Indicadores
    ema_f       = ema(prices, config.EMA_FAST)
    ema_s       = ema(prices, config.EMA_SLOW)
    ema_200_val = ema(prices, 200)
    ema_20_val  = ema(prices, 20)
    rsi_vals    = rsi(prices, config.RSI_PERIOD)
    _, _, macd_hist = macd(prices)
    atr_vals    = atr(df)
    vwap_vals   = vwap(df)
    zscore_vals = zscore_vwap(df)
    adx_vals    = adx(df, 14)
    
    # 2. Patrones y Filtros
    is_vol_spike = bool(volume_spike(df).iloc[-1])
    is_hammer    = bool(detect_hammer(df).iloc[-1])
    is_engulf    = bool(detect_engulfing(df).iloc[-1])
    
    # 3. Valores actuales
    close_now   = float(prices.iloc[-1])
    ema_f_now   = float(ema_f.iloc[-1])
    ema_s_now   = float(ema_s.iloc[-1])
    ema_20_now  = float(ema_20_val.iloc[-1])
    ema_200_now = float(ema_200_val.iloc[-1])
    rsi_now     = float(rsi_vals.iloc[-1])
    macd_now    = float(macd_hist.iloc[-1])
    macd_prev   = float(macd_hist.iloc[-2])
    atr_now     = float(atr_vals.iloc[-1])
    vwap_now    = float(vwap_vals.iloc[-1])
    zscore_now  = float(zscore_vals.iloc[-1]) if not pd.isna(zscore_vals.iloc[-1]) else 0.0
    adx_now     = float(adx_vals.iloc[-1]) if not pd.isna(adx_vals.iloc[-1]) else 0.0
    
    # 4. Régimen de Mercado
    vol_avg   = float(df["Volume"].rolling(20).mean().iloc[-1])
    vol_ratio = float(df["Volume"].iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
    atr_pct   = (atr_now / close_now) * 100 if close_now else 0.0
    
    current_regime = detect_regime(adx_now, atr_pct, close_now, ema_200_now, rsi_now, vol_ratio)
    regime_label   = REGIMEN_LABELS.get(current_regime, current_regime)
    
    log.info(f"{sym_prefix}🌍 RÉGIMEN: {regime_label} | ADX={adx_now:.1f} | ATR%={atr_pct:.2f}% | Vol={vol_ratio:.1f}x")

    # 5. Lógica de Estrategias (Simplified for readability after refactor)
    signal = SIGNAL_HOLD
    confirmations = []
    blocks = []
    
    # ESTRATEGIA A: Oliver Vélez (Rebote 20)
    is_velez = (ema_20_now > ema_200_now) and (df["Low"].iloc[-1] <= ema_20_now * 1.002) and (close_now > ema_20_now) and (close_now > df["Open"].iloc[-1])
    
    # ESTRATEGIA B: VWAP Bounce
    is_vwap_bounce = (zscore_now < -2.0) and (close_now > df["Open"].iloc[-1] or is_hammer) and (close_now > ema_200_now)
    
    # ESTRATEGIA C: Bollinger Reversal
    _, _, bb_low = bollinger_bands(prices)
    is_bb_rev = (df["Low"].iloc[-1] <= bb_low.iloc[-1]) and (close_now > df["Open"].iloc[-1])
    
    # ESTRATEGIA D: EMA Crossover
    is_ema_x = (ema_f.iloc[-2] < ema_s.iloc[-2]) and (ema_f_now > ema_s_now) and (35 < rsi_now < 70)

    # Evaluación por Régimen
    is_inverted = asset_type == "inverted"
    eff_regime = current_regime
    if is_inverted:
        eff_regime = "TREND_UP" if current_regime == "TREND_DOWN" else "TREND_DOWN" if current_regime == "TREND_UP" else current_regime

    if eff_regime in ("TREND_UP", "MOMENTUM", "NEUTRAL") and is_velez:
        signal = SIGNAL_BUY
        confirmations.append(f"🎯 [{regime_label}] Oliver Vélez (EMA 20 Bounce)")
    
    elif eff_regime in ("RANGE", "NEUTRAL"):
        if is_vwap_bounce:
            signal = SIGNAL_BUY
            confirmations.append(f"🎯 [{regime_label}] VWAP Bounce (Z={zscore_now:.2f})")
        elif is_bb_rev:
            signal = SIGNAL_BUY
            confirmations.append(f"🎯 [{regime_label}] Bollinger Reversal (Lower Band)")
        elif is_ema_x:
            signal = SIGNAL_BUY
            confirmations.append(f"🎯 [{regime_label}] EMA Crossover {config.EMA_FAST}/{config.EMA_SLOW}")

    # Flash Crash / Capitulación
    if eff_regime in ("TREND_DOWN", "CHAOS") and zscore_now < -2.5 and (is_hammer or is_engulf or is_bb_rev):
        signal = SIGNAL_BUY
        confirmations.append(f"🔥 [{regime_label}] Capitulación (Z={zscore_now:.2f})")

    # Filtros de Calidad
    if signal == SIGNAL_BUY:
        if (43 < rsi_now < 47 or adx_now < 10) and zscore_now > -2.3:
            log.warning(f"{sym_prefix}🛑 Bloqueado por RSI Zona Muerta o ADX bajo.")
            signal = SIGNAL_HOLD
            confirmations.clear()
            blocks.append("Filtro de calidad (RSI/ADX)")


    # Venta de emergencia
    if signal == SIGNAL_HOLD and close_now < ema_20_now and macd_now < macd_prev and macd_now < 0:
        signal = SIGNAL_SELL
        confirmations.append("🛡️ Venta forzada: pérdida de estructura")

    # ML Features
    _vwap_dist = (close_now - vwap_now) / vwap_now * 100 if vwap_now > 0 else 0.0
    _hour_ny = float(df.index[-1].hour) if hasattr(df.index[-1], 'hour') else 10.0
    ml_features = {
        'symbol': symbol,
        'hour_of_day': _hour_ny,
        'vwap_dist_pct': _vwap_dist,
        'rsi': rsi_now, 'macd_hist': macd_now, 
        'ema_diff_pct': (ema_f_now - ema_s_now) / close_now * 100,
        'atr_pct': atr_pct, 'adx': adx_now, 'regime': current_regime,
        'vol_ratio': vol_ratio, 'zscore_vwap': zscore_now,
        'num_confirmations': len(confirmations),
        'has_pattern': is_hammer or is_engulf or is_bb_rev,
        'is_adx_rising': adx_now > adx_vals.iloc[-2]
    }

    # AI Check
    if signal == SIGNAL_BUY:
        ai_model = get_ai_model()
        if ai_model:
            try:
                rf_decision = ai_model.predict(pd.DataFrame([ml_features])[ai_model.feature_names_in_])[0]
                if rf_decision == 0:
                    signal = SIGNAL_HOLD
                    confirmations.clear()
                    blocks.append("🤖 RandomForest bloqueó la entrada")
            except: pass

        # Neural Filter
        try:
            from shared.utils.neural_filter import get_neural_filter
            proba, _ = get_neural_filter().predict(get_neural_filter().build_features(**ml_features))
            if proba < config.CONFIDENCE_THRESHOLD:
                signal = SIGNAL_HOLD
                confirmations.clear()
                blocks.append(f"🧠 NeuralFilter: Probabilidad baja ({proba*100:.1f}%)")
        except: pass

    return SignalResult(
        signal=signal, ema_fast=ema_f_now, ema_slow=ema_s_now, ema_200=ema_200_now,
        rsi_value=rsi_now, macd_hist=macd_now, vwap_value=vwap_now, atr_value=atr_now,
        close=close_now, timestamp=df.index[-1], confirmations=confirmations, blocks=blocks,
        ml_features=ml_features, regime=current_regime
    )

