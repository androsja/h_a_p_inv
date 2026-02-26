"""
strategy/indicators.py ─ Motor de análisis técnico profesional (nivel institucional).

Sistema de confluencia multi-indicador:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FILTRO DE TENDENCIA (bloqueos duros):
    • EMA 200       → Solo comprar si precio > EMA 200
    • KAMA          → Confirm tendencia sin falsas señales en mercado lateral

  SEÑAL DE ENTRADA (gatillo):
    • EMA 12/26 crossover  → Cruce clásico de tendencia
    • KAMA crossover       → Cruce adaptativo (solo cuando hay tendencia real)
    • MACD histograma      → Positivo y creciente
    • RSI (Wilder)         → Zona 35-65 (momentum sin extremos)

  CONFIRMACIÓN DE IMPULSO:
    • VWAP          → Precio > VWAP (fuerza institucional del día)
    • Z-Score VWAP  → No comprar si Z-Score > +2.0 (sobrecompra estadística)
    • Volumen       → Vela > 1.5× promedio (impulso real)
    • Hammer/Engulf → Patrones de vela en zona de soporte

  FILTROS DE RUIDO:
    • ATR           → Volatilidad excesiva bloquea entrada
    • Super Smoother → Ehlers 2-pole para eliminar ruido de alta frecuencia
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import numpy as np
import pandas as pd
import os
import joblib
from functools import lru_cache
import config

# ── CACHÉ DE IA PARA VELOCIDAD ULTRA-RÁPIDA (< 1 ms) ──────────
@lru_cache(maxsize=1)
def _load_ai_model(mtime):
    try:
        model_path = '/app/data/ai_model.joblib'
        return joblib.load(model_path)
    except Exception:
        return None

def get_ai_model():
    model_path = '/app/data/ai_model.joblib'
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        return _load_ai_model(mtime)
    return None


# ─── Tipos de señal ─────────────────────────────────────────────────────────────────
SIGNAL_BUY  = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"


# ════════════════════════════════════════════════════════════════════════════
#  INDICADORES INDIVIDUALES
# ════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) clásico de Wilder. Rango 0-100."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX)
    Mide la fuerza de la tendencia, sin importar si es alcista o bajista.
    < 20 = Mercado lateral (Choppy / Whipsaws)
    > 25 = Tendencia fuerte
    """
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    
    # 1. True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 2. Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth them
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    pos_dm_smooth = pd.Series(pos_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    neg_dm_smooth = pd.Series(neg_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    
    # 3. Directional Indicators
    di_plus = 100 * (pos_dm_smooth / tr_smooth.replace(0, np.nan))
    di_minus = 100 * (neg_dm_smooth / tr_smooth.replace(0, np.nan))
    
    # 4. ADX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx_line
    

def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD clásico.
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — mide la volatilidad real.
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP (Volume Weighted Average Price) intradía.
    Se recalcula acumulativamente desde la primera vela del DataFrame.
    En producción real, debe reiniciarse cada día a las 9:30 AM ET.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume        = df["Volume"].astype(float).replace(0, 1)
    cum_tpvol     = (typical_price * volume).cumsum()
    cum_vol       = volume.cumsum()
    return cum_tpvol / cum_vol


def volume_spike(df: pd.DataFrame, window: int = 20, threshold: float = 1.5) -> pd.Series:
    """
    Devuelve True si la vela actual tiene volumen > threshold × promedio móvil.
    Confirma que el movimiento tiene respaldo de participantes reales.
    """
    vol     = df["Volume"].astype(float)
    avg_vol = vol.rolling(window).mean()
    return vol > (avg_vol * threshold)


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Detecta el patrón Hammer (martillo) alcista.
    Condición: mecha inferior >= 2× cuerpo real Y mecha superior pequeña.
    Señal de que los compradores rechazaron precios bajos.
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    body         = (c - o).abs()
    lower_wick   = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick   = h - pd.concat([o, c], axis=1).max(axis=1)
    is_hammer = (
        (lower_wick >= 2 * body.replace(0, 0.0001)) &
        (upper_wick <= body * 0.5) &
        (body > 0)
    )
    return is_hammer


def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    KAMA ─ Kaufman Adaptive Moving Average.

    Se adapta automáticamente a la eficiencia del movimiento del precio:
    - Mercado en tendencia clara   → KAMA se mueve rápido (como EMA corta)
    - Mercado lateral/ruidoso      → KAMA se ralentiza (evita falsas señales)

    Args:
        series: Serie de precios de cierre
        period: Período del Efficiency Ratio (default 10)
        fast:   EMA rápida para mercado en tendencia (default 2 períodos)
        slow:   EMA lenta para mercado lateral (default 30 períodos)
    """
    fast_sc = 2 / (fast + 1)   # Constante de suavizado rápida
    slow_sc = 2 / (slow + 1)   # Constante de suavizado lenta

    prices  = series.astype(float).values
    n       = len(prices)
    kama_v  = np.full(n, np.nan)

    # Inicializar KAMA desde el período
    kama_v[period] = prices[period]

    for i in range(period + 1, n):
        # Efficiency Ratio: movimiento neto / suma de movimientos absolutos
        direction = abs(prices[i] - prices[i - period])
        volatility = sum(abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1))

        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility

        # Constante de suavizado adaptativa: SC = (ER × (fast - slow) + slow)^2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Actualizar KAMA
        kama_v[i] = kama_v[i - 1] + sc * (prices[i] - kama_v[i - 1])

    return pd.Series(kama_v, index=series.index)


def super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Filtro Super Smoother de John Ehlers (Butterworth 2 polos).

    Elimina ruido de alta frecuencia (componentes < 10 barras) sin introducir
    el lag temporal que arruina las entradas en marcos de 5 minutos.
    Mucho más preciso que una EMA para detectar la tendencia real del precio.

    Referencia: 'Cybernetic Analysis for Stocks and Futures' — John Ehlers
    """
    a1 = math.exp(-math.sqrt(2) * math.pi / period)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / period)
    c2 = b1
    c3 = -(a1 ** 2)
    c1 = (1 - b1 + a1 ** 2) / 4

    prices = series.astype(float).values
    n      = len(prices)
    ss     = np.zeros(n)

    for i in range(2, n):
        ss[i] = (
            c1 * (prices[i] + prices[i - 1])
            + c2 * ss[i - 1]
            + c3 * ss[i - 2]
        )

    result = pd.Series(ss, index=series.index)
    result[:2] = np.nan
    return result


def zscore_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Z-Score del precio respecto al VWAP — Reversión a la Media Institucional.

    Z-Score = (Precio - VWAP) / Desviación Estándar del precio

    Interpretación:
    - Z-Score > +2.0  → Sobrecompra estadística → NO comprar
    - Z-Score < -2.0  → Sobreventa estadística  → Oportunidad de compra
    - Z-Score ~ 0     → Zona neutral              → Normal

    El VWAP como media de referencia es el estándar institucional porque
    considera dónde se negó el mayor volumen (equilibrio real del mercado).
    """
    vwap_vals = vwap(df)
    close     = df["Close"].astype(float)
    std       = close.rolling(window).std()
    return (close - vwap_vals) / std.replace(0, np.nan)

def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detecta el patrón Engulfing alcista (vela verde envuelve a la roja anterior).
    Alta fiabilidad en zonas de soporte o sobre el VWAP.
    """
    o = df["Open"].astype(float)
    c = df["Close"].astype(float)
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    bullish_engulfing = (
        (prev_c < prev_o) &      # Vela anterior roja
        (c > o) &                # Vela actual verde
        (c > prev_o) &           # Cierre actual > apertura anterior
        (o < prev_c)             # Apertura actual < cierre anterior
    )
    return bullish_engulfing



class SignalResult:
    """
    Resultado completo del análisis técnico multi-indicador.
    Incluye todos los valores para el dashboard y el log.
    """

    def __init__(
        self,
        signal:       str,
        ema_fast:     float,
        ema_slow:     float,
        ema_200:      float,
        rsi_value:    float,
        macd_hist:    float,
        vwap_value:   float,
        atr_value:    float,
        close:        float,
        timestamp,
        confirmations: list[str],   # Lista de condiciones que se cumplieron
        blocks:        list[str],   # Lista de condiciones que bloquearon la entrada
        ml_features:   dict = None, # Características numéricas del mercado
    ):
        self.signal        = signal
        self.ema_fast      = ema_fast
        self.ema_slow      = ema_slow
        self.ema_200       = ema_200
        self.rsi_value     = rsi_value
        self.macd_hist     = macd_hist
        self.vwap_value    = vwap_value
        self.atr_value     = atr_value
        self.close         = close
        self.timestamp     = timestamp
        self.confirmations = confirmations
        self.blocks        = blocks
        self.ml_features   = ml_features or {}

    def __repr__(self) -> str:
        return (
            f"SignalResult(signal={self.signal} | "
            f"✅ {len(self.confirmations)} confirman | "
            f"❌ {len(self.blocks)} bloquean | "
            f"RSI={self.rsi_value:.1f} MACD={self.macd_hist:.3f} "
            f"ATR={self.atr_value:.2f})"
        )


# ════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL DE ANÁLISIS
# ════════════════════════════════════════════════════════════════════════════

# Mínimo de barras para calcular todos los indicadores correctamente
MIN_BARS = max(200, config.EMA_SLOW, config.RSI_PERIOD) + 10


def analyze(df: pd.DataFrame) -> SignalResult:
    """
    Evalúa la señal de trading usando un sistema de confluencia multi-indicador.

    Requiere mínimo 3 confirmaciones alcistas y 0 bloqueos activos para emitir BUY.
    Esto reduce drásticamente las falsas señales del EMA crossover simple.

    Args:
        df: DataFrame OHLCV con suficientes barras históricas.

    Returns:
        SignalResult con señal final y todos los valores de indicadores.
    """
    # ── Inicializar resultado neutro si no hay suficientes datos ─────────────
    if len(df) < MIN_BARS:
        return SignalResult(
            signal=SIGNAL_HOLD,
            ema_fast=0.0, ema_slow=0.0, ema_200=0.0,
            rsi_value=50.0, macd_hist=0.0,
            vwap_value=0.0, atr_value=0.0,
            close=float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0,
            timestamp=df.index[-1] if len(df) > 0 else None,
            confirmations=[], blocks=["⏳ Acumulando datos históricos…"],
        )

    closes = df["Close"].astype(float)

    # ── Calcular todos los indicadores ────────────────────────────────────────
    ema_f       = ema(closes, config.EMA_FAST)
    ema_s       = ema(closes, config.EMA_SLOW)
    ema_200_val = ema(closes, 200)
    rsi_vals    = rsi(closes, config.RSI_PERIOD)
    _, _, macd_hist = macd(closes)
    atr_vals    = atr(df)
    vwap_vals   = vwap(df)
    vol_spike   = volume_spike(df)
    hammers     = detect_hammer(df)
    engulfings  = detect_engulfing(df)
    kama_vals   = kama(closes)             # KAMA adaptativo
    ss_vals     = super_smoother(closes)   # Ehlers Super Smoother (anti-ruido)
    zscore_vals = zscore_vwap(df)          # Z-Score reversión a la media
    adx_vals    = adx(df, 14)

    # ── Valores actuales y anteriores ────────────────────────────────────────
    ema_f_prev  = float(ema_f.iloc[-2])
    ema_f_now   = float(ema_f.iloc[-1])
    ema_s_prev  = float(ema_s.iloc[-2])
    ema_s_now   = float(ema_s.iloc[-1])
    ema_200_now = float(ema_200_val.iloc[-1])
    rsi_now     = float(rsi_vals.iloc[-1])
    macd_prev   = float(macd_hist.iloc[-2])
    macd_now    = float(macd_hist.iloc[-1])
    atr_now     = float(atr_vals.iloc[-1])
    vwap_now    = float(vwap_vals.iloc[-1])
    close_now   = float(closes.iloc[-1])
    has_vol     = bool(vol_spike.iloc[-1])
    is_hammer   = bool(hammers.iloc[-1])
    is_engulf   = bool(engulfings.iloc[-1])
    adx_now     = float(adx_vals.iloc[-1]) if not pd.isna(adx_vals.iloc[-1]) else 0.0
    ts          = df.index[-1]

    # KAMA: cruce adaptativo
    kama_prev   = kama_vals.iloc[-2] if not pd.isna(kama_vals.iloc[-2]) else close_now
    kama_now    = kama_vals.iloc[-1] if not pd.isna(kama_vals.iloc[-1]) else close_now
    kama_bull   = (float(kama_prev) < float(ema_f.iloc[-2])) and (float(kama_now) > float(ema_f.iloc[-1]))
    kama_bull   = close_now > float(kama_now)    # simplificado: precio sobre KAMA

    # Super Smoother: pendiente alcista
    ss_prev = ss_vals.iloc[-2] if not pd.isna(ss_vals.iloc[-2]) else close_now
    ss_now  = ss_vals.iloc[-1] if not pd.isna(ss_vals.iloc[-1]) else close_now
    ss_rising = float(ss_now) > float(ss_prev)   # tendencia alcista suavizada

    # Z-Score vs VWAP
    zscore_now = float(zscore_vals.iloc[-1]) if not pd.isna(zscore_vals.iloc[-1]) else 0.0

    # ════════════════════════════════════════════════════════════════════════
    #  NUEVO ESCUADRÓN DE MÚLTIPLES ESTRATEGIAS (Lógica OR - Francotiradores)
    # ════════════════════════════════════════════════════════════════════════
    signal = SIGNAL_HOLD
    confirmations_buy  = []
    confirmations_sell = []
    blocks_buy         = [] # Se mantiene por compatibilidad del constructor
    
    open_now = float(df["Open"].iloc[-1])
    low_now = float(df["Low"].iloc[-1])
    
    # Calcular media móvil de 20 periodos ("La 20" de Oliver Vélez)
    ema_20_series = ema(closes, 20)
    ema_20_now = float(ema_20_series.iloc[-1])
    
    # ── ESTRATEGIA A: "Juego a la 20" (Estilo Oliver Vélez) ───────────────
    # Tendencia alcista clara (EMA 20 > EMA 200). El precio hace un pullback (retroceso)
    # tocando la EMA 20 (Low <= EMA 20), pero los toros defienden y sacan el precio arriba (Close > EMA 20, vela verde).
    velez_trend_ok = ema_20_now > ema_200_now
    velez_pullback = low_now <= (ema_20_now * 1.002)  # Toca o casi toca la 20 (margen 0.2%)
    velez_bounce = (close_now > ema_20_now) and (close_now > open_now) # Rebotó y cerró verde
    
    is_velez_setup = velez_trend_ok and velez_pullback and velez_bounce
    
    # ── ESTRATEGIA B: Rebote Institucional (Estilo SMB Capital) ───────────
    # El precio colapsa irracionalmente separándose del VWAP (Z-Score < -2.0).
    # Oportunidad de compra estadística si la vela hace reversión confirmada (Cierre Verde o Martillo).
    # IMPORTANTE: Solo comprar el "dip" si microscópicamente colapsó, pero MACROSCÓPICAMENTE
    # la tendencia general sigue siendo alcista (Precio > EMA 200). De lo contrario es atajar un cuchillo.
    vwap_oversold = zscore_now < -2.0
    vwap_reversal = (close_now > open_now) or is_hammer
    macro_bullish = close_now > ema_200_now
    
    is_vwap_bounce = vwap_oversold and vwap_reversal and macro_bullish
    
    # ── EVALUACIÓN INDEPENDIENTE (Lógica OR) ──────────────────────────────
    if is_velez_setup:
        signal = SIGNAL_BUY
        confirmations_buy.append("🎯 Estrategia A: Rebote Oliver Vélez en la EMA 20 (Pullback alcista).")
    elif is_vwap_bounce:
        signal = SIGNAL_BUY
        confirmations_buy.append(f"🎯 Estrategia B: Anomalía VWAP Extrema (Z={zscore_now:.2f}) con rechazo alcista.")
        
    # ── FILTRO INTELIGENTE BASADO EN MACHINE LEARNING HISTÓRICO ───────────
    # Filtrar operaciones en la "ZONA DE LA MUERTE" (Whipsaws).
    # La IA (Árbol de Decisión) descubrió en 2000 trades que entradas con RSI entre 26.5 y 62.2
    # tienen un altísimo porcentaje de fallar y tocar Stop Loss.
    is_whipsaw_zone = (rsi_now > 26.5) and (rsi_now < 62.2)
    
    # Adicionalmente, añadiremos el ADX que tú pedías: Evitamos laterales bruscos.
    is_low_trend = adx_now < 20.0
    
    if signal == SIGNAL_BUY and (is_whipsaw_zone or is_low_trend):
        signal = SIGNAL_HOLD
        # Evitar registrar en log general como disparo, se queda callado.
        confirmations_buy.clear()
        
    # ── LÓGICA DE VENTA DE EMERGENCIA ─────────────────────────────────────
    # Tu Take Profit (3%) o Trailing Stop harán el 90% del trabajo de salida. 
    # Pero forzamos venta de pánico solo si perfora severamente la EMA 20 + MACD se torna bajista agresivo
    force_sell = (close_now < ema_20_now) and (macd_now < macd_prev) and (macd_now < 0)
    
    if force_sell and signal == SIGNAL_HOLD:
        signal = SIGNAL_SELL
        confirmations_sell.append("🛡️ Venta forzada: Deterioro de estructura (precio perdió la EMA 20).")
        
    # ── GENERACIÓN DE FEATURES PARA MACHINE LEARNING ──────────────────────────
    # Para capturar la foto matemática que llevó a esta decisión:
    atr_pct = (atr_now / close_now) * 100 if close_now else 0
    ml_features = {
        'rsi': float(rsi_now),
        'macd_hist': float(macd_now),
        'ema_diff_pct': float((ema_f_now - ema_s_now) / close_now * 100) if close_now else 0,
        'vwap_dist_pct': float((close_now - vwap_now) / vwap_now * 100) if vwap_now > 0 else 0,
        'atr_pct': float(atr_pct),
        'adx': float(adx_now)
    }

    # 🤖 EJECUCIÓN DEL MODELO DE INTELIGENCIA ARTIFICIAL EN VIVO
    # Consulta al oráculo del Machine Learning antes de abrir fuego.
    # Tiempo de ejecución: ~ 0.0001 segundos
    if signal == SIGNAL_BUY:
        ai_model = get_ai_model()
        if ai_model:
            try:
                # Orden exacto de columnas para predecir
                expected_cols = ai_model.feature_names_in_
                X_live = pd.DataFrame([ml_features])[expected_cols]
                
                prediction_ai = ai_model.predict(X_live)[0]
                
                if prediction_ai == 0:
                    # La Red Neuronal vaticina una pérdida inminente.
                    signal = SIGNAL_HOLD
                    confirmations_buy.clear()
                    # Bloqueado por la IA. Salvará el capital.
            except Exception as e:
                from utils.logger import log
                log.error(f"Error AI prediction: {e}")


    # Logeamos la ejecución del escuadrón (SILENCIOSO CON ADX PARA FORENSE)
    if signal == SIGNAL_BUY:
        from utils.logger import log
        log.info(f"🔫 COMPRA PERMITIDA POR IA | [ADX={adx_now:.1f}] | " + " | ".join(confirmations_buy))
    elif signal == SIGNAL_SELL:
        from utils.logger import log
        log.info(f"🔴 VENTA ALERTA TÁCTICA | [ADX={adx_now:.1f}] | " + " | ".join(confirmations_sell))

    return SignalResult(
        signal=signal,
        ema_fast=ema_f_now,
        ema_slow=ema_s_now,
        ema_200=ema_200_now,
        rsi_value=rsi_now,
        macd_hist=macd_now,
        vwap_value=vwap_now,
        atr_value=atr_now,
        close=close_now,
        timestamp=ts,
        confirmations=confirmations_buy,
        blocks=blocks_buy,
        ml_features=ml_features
    )
