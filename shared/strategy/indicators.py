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
from typing import Optional, Union, Dict, List, Tuple

from shared import config
from shared.utils.logger import log

# ── Importar módulos especializados ───────────────────────────────────────────
from shared.strategy.indicators_calc import (
    ema, rsi, adx, macd, atr, vwap, zscore_vwap, bollinger_bands, kama, super_smoother, choppiness_index
)
from shared.strategy.patterns import volume_spike, detect_hammer, detect_engulfing
from shared.strategy.regimes  import detect_regime, REGIMEN_LABELS
from shared.strategy.signal_models import SignalResult
from shared.strategy.ml_predictor import ml_predictor

# Constantes de señal
SIGNAL_BUY  = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"

MIN_BARS = config.MIN_BARS_REQUIRED

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

_regime_buffer: Dict[str, Tuple[Optional[str], float]] = {}  # {symbol: (last_label, last_time)}

def analyze(df: pd.DataFrame, symbol: str = "", asset_type: str = "normal") -> SignalResult:
    sym_prefix = f"[{symbol}] " if symbol else ""
    
    if len(df) < MIN_BARS:
        if symbol:
            log.warning(f"{sym_prefix}⏳ Datos insuficientes: {len(df)}/{MIN_BARS} barras.")
        return SignalResult(
            signal=SIGNAL_HOLD,
            ema_fast=0.0, ema_slow=0.0, ema_200=0.0, rsi_value=50.0, macd_hist=0.0,
            vwap_value=0.0, atr_value=0.0, chop_value=50.0,
            close=float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0,
            timestamp=df.index[-1] if len(df) > 0 else None,
            blocks=[f"⏳ Cargando datos ({len(df)}/{MIN_BARS})"],
        )

    prices = df["Close"].astype(float)
    
    # 1. Calcular Indicadores
    ema_f       = ema(prices, config.EMA_FAST)
    ema_s       = ema(prices, config.EMA_SLOW)
    ema_200_val = ema(prices, config.EMA_LONG_PERIOD)
    ema_20_val  = ema(prices, config.EMA_MEDIUM_PERIOD)
    rsi_vals    = rsi(prices, config.RSI_PERIOD)
    _, _, macd_hist = macd(prices)
    atr_vals    = atr(df)
    vwap_vals   = vwap(df)
    zscore_vals = zscore_vwap(df)
    adx_vals    = adx(df, config.ADX_PERIOD)
    chop_vals   = choppiness_index(df, config.CHOP_PERIOD)
    
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
    ema_200_prev= float(ema_200_val.iloc[-2]) if len(ema_200_val) > 1 else ema_200_now
    rsi_now     = float(rsi_vals.iloc[-1])
    macd_now    = float(macd_hist.iloc[-1])
    macd_prev   = float(macd_hist.iloc[-2])
    atr_now     = float(atr_vals.iloc[-1])
    vwap_now    = float(vwap_vals.iloc[-1])
    zscore_now  = float(zscore_vals.iloc[-1]) if not pd.isna(zscore_vals.iloc[-1]) else 0.0
    adx_now     = float(adx_vals.iloc[-1]) if not pd.isna(adx_vals.iloc[-1]) else 0.0
    chop_now    = float(chop_vals.iloc[-1]) if not pd.isna(chop_vals.iloc[-1]) else 50.0
    
    # 4. Régimen de Mercado
    vol_avg   = float(df["Volume"].rolling(config.VOLUME_ROLLING_WIN).mean().iloc[-1])
    vol_ratio = float(df["Volume"].iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
    atr_pct   = (atr_now / close_now) * 100 if close_now else 0.0
    
    current_regime = detect_regime(adx_now, atr_pct, close_now, ema_200_now, rsi_now, vol_ratio)
    regime_label   = REGIMEN_LABELS.get(current_regime, current_regime)
    
    # [OPTIMIZACIÓN LOGS] Solo loguear régimen si cambia y han pasado al menos 10s (tiempo real)
    import time
    now_real = time.time()
    last_r, last_t = _regime_buffer.get(symbol, (None, 0))
    
    if (last_r != regime_label) and (now_real - last_t > config.REGIME_LOG_INTERVAL):
        log.info(f"{sym_prefix}🌍 RÉGIMEN: {regime_label} | ADX={adx_now:.1f} | ATR%={atr_pct:.2f}% | Vol={vol_ratio:.1f}x")
        _regime_buffer[symbol] = (regime_label, now_real)

    # 5. Lógica de Estrategias
    signal = SIGNAL_HOLD
    confirmations = []
    blocks = []
    prob_win = 0.5
    is_ml_blocked = False
    is_quality_blocked = False
    
    # ESTRATEGIA A: Oliver Vélez (Rebote 20)
    is_velez = (ema_20_now > ema_200_now) and (df["Low"].iloc[-1] <= ema_20_now * config.VELEZ_BOUNCE_MULT) and (close_now > ema_20_now) and (close_now > df["Open"].iloc[-1])
    
    # ESTRATEGIA B: VWAP Bounce
    is_vwap_bounce = (zscore_now < config.VWAP_BOUNCE_ZSCORE) and (close_now > df["Open"].iloc[-1] or is_hammer) and (close_now > ema_200_now)
    
    # ESTRATEGIA C: Bollinger Reversal
    _, _, bb_low = bollinger_bands(prices)
    is_bb_rev = (df["Low"].iloc[-1] <= bb_low.iloc[-1]) and (close_now > df["Open"].iloc[-1])
    
    # ESTRATEGIA D: EMA Crossover
    is_ema_x = (ema_f.iloc[-2] < ema_s.iloc[-2]) and (ema_f_now > ema_s_now) and (config.EMA_CROSS_RSI_MIN < rsi_now < config.EMA_CROSS_RSI_MAX)
    
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
    if eff_regime in ("TREND_DOWN", "CHAOS") and zscore_now < config.EMERGENCY_ZSCORE and (is_hammer or is_engulf or is_bb_rev):
        signal = SIGNAL_BUY
        confirmations.append(f"🔥 [{regime_label}] Capitulación (Z={zscore_now:.2f})")



    # Venta de emergencia
    if signal == SIGNAL_HOLD and close_now < ema_20_now and macd_now < macd_prev and macd_now < 0:
        signal = SIGNAL_SELL
        confirmations.append("🛡️ Venta forzada: pérdida de estructura")

    # [BLOQUEO TOTAL] Filtros de Calidad Académicos (Anti-Whipsaw)
    is_quality_blocked = False
    
    # 1. Bloqueo por falta de fuerza (ADX)
    if adx_now < config.QUALITY_ADX_THRESHOLD:
        is_quality_blocked = True
        blocks.append(f"🛑 Fuerza insuficiente (ADX={adx_now:.1f} < {config.QUALITY_ADX_THRESHOLD})")
    
    # 2. Bloqueo por lateralidad (Choppiness)
    if chop_now > config.QUALITY_CHOP_THRESHOLD:
        is_quality_blocked = True
        blocks.append(f"🛑 Mercado lateral (CHOP={chop_now:.1f} > {config.QUALITY_CHOP_THRESHOLD})")

    # 3. Filtro de tendencia mayor (Daily/Hourly structure) y Pendiente
    if not is_inverted and signal == SIGNAL_BUY:
        if close_now < ema_200_now:
            is_quality_blocked = True
            blocks.append(f"🛑 Bajo EMA 200 (Tendencia Bajista)")
        elif ema_200_now <= ema_200_prev:
            is_quality_blocked = True
            blocks.append(f"🛑 EMA 200 Plana o Descendente (Sin fuerza)")

    # ML Features
    _vwap_dist = (close_now - vwap_now) / vwap_now * 100 if vwap_now > 0 else 0.0
    _hour_ny = float(df.index[-1].hour) if hasattr(df.index[-1], 'hour') else 10.0
    ml_features = {
        'symbol': symbol,
        'hour_of_day': _hour_ny,
        'vwap_dist_pct': _vwap_dist,
        'rsi': rsi_now, 'macd_hist': macd_now, 
        'ema_diff_pct': (ema_f_now - ema_s_now) / close_now * 100,
        'atr_pct': atr_pct, 'adx': adx_now, 'chop': chop_now,
        'regime': current_regime,
        'vol_ratio': vol_ratio, 'zscore_vwap': zscore_now,
        'num_confirmations': len(confirmations),
        'has_pattern': is_hammer or is_engulf or is_bb_rev,
        'is_adx_rising': adx_now > adx_vals.iloc[-2],
        'model_accuracy': getattr(ml_predictor, 'accuracy', 0.0),
        'total_samples': ml_predictor.get_sample_count()
    }

    # Reset signal if blocked
    if is_quality_blocked:
        signal = SIGNAL_HOLD

    # AI Check (Solo si pasó los filtros de calidad)
    is_ml_blocked = False
    prob_win_rf = 0.5
    prob_win_nf = 0.5
    
    if signal == SIGNAL_BUY:
        # 1. Alpha-Optimizer (RF Classifier + Regressor)
        is_alpha, prob_win_rf, expected_pnl = ml_predictor.predict_win(ml_features, current_regime)

        # 2. Neural Filter
        try:
            from shared.utils.neural_filter import get_neural_filter
            nf = get_neural_filter(symbol)
            nf_features = nf.build_features(
                symbol=symbol,
                hour_of_day=ml_features.get('hour_of_day', 10.0),
                vwap_dist_pct=ml_features.get('vwap_dist_pct', 0.0),
                rsi=ml_features.get('rsi', 50.0), 
                macd_hist=ml_features.get('macd_hist', 0.0),
                atr_pct=ml_features.get('atr_pct', 0.0), 
                vol_ratio=ml_features.get('vol_ratio', 1.0),
                ema_spread_pct=ml_features.get('ema_diff_pct', 0.0),
                zscore_vwap=zscore_now, 
                regime=current_regime,
                num_confirmations=len(confirmations),
                adx=adx_now,
                has_pattern=ml_features.get('has_pattern', False),
                is_adx_rising=ml_features.get('is_adx_rising', False)
            )
            prob_win_nf, _ = nf.predict(nf_features)
        except Exception as e:
            log.error(f"Error en Neural Filter: {e}")
            prob_win_nf = 0.5

        # 🚀 Lógica de "Consenso Inteligente":
        # Bloqueamos si el Alpha-Optimizer lo rechaza (is_alpha=False)
        # O si hay un rechazo unánime por probabilidad.
        threshold_strict = 0.45
        threshold_balanced = config.CONFIDENCE_THRESHOLD # 0.55
        
        # Un trade es rechazado por la IA si no es Alpha o si las probabilidades son muy bajas
        if not is_alpha or prob_win_nf < threshold_strict:
            is_ml_blocked = True
            reason = "BAJO VALOR" if not is_alpha else "PROB BAJA"
            blocks.append(f"🤖 Alpha Block: {reason} | RF={prob_win_rf:.0%} | PnL_Est=${expected_pnl:.2f}")
        
        prob_win = (prob_win_rf + prob_win_nf) / 2.0

    # Log de depuración final (solo en modo DEBUG para no frenar la simulación)
    # Generar Recomendación Verbal Didáctica
    ai_rec = "Analizando mercado..."
    if regime_label == "CAOS":
        ai_rec = "☢️ MERCADO EN CAOS: Mejor observar desde afuera."
    elif signal == SIGNAL_BUY:
        if prob_win > 0.75: ai_rec = "🧠 IA SUGIERE: Compra con ALTA confianza."
        elif prob_win > 0.60: ai_rec = "📈 IA COINCIDE: Estructura técnica sólida."
        else: ai_rec = "⚖️ IA EN DUDA: Señal técnica débil."
    else:
        if regime_label == "TREND_UP": ai_rec = "🌤️ TENDENCIA ALCISTA: Buscando retroceso."
        elif regime_label == "TREND_DOWN": ai_rec = "⛈️ TENDENCIA BAJISTA: Vigilando rebote."
        else: ai_rec = "💤 ESPERANDO OPORTUNIDAD: Sin señales claras."

    return SignalResult(
        signal=signal, ema_fast=ema_f_now, ema_slow=ema_s_now, ema_200=ema_200_now,
        rsi_value=rsi_now, macd_hist=macd_now, vwap_value=vwap_now, atr_value=atr_now,
        chop_value=chop_now,
        close=close_now, timestamp=df.index[-1], confirmations=confirmations, blocks=blocks,
        ml_features=ml_features, regime=current_regime, is_quality_blocked=is_quality_blocked,
        is_ml_blocked=is_ml_blocked, ai_win_prob=round(prob_win if signal == SIGNAL_BUY else 0.5, 2),
        ai_recommendation=ai_rec
    )
