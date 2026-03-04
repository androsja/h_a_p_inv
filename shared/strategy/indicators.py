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
from shared import config

# ── CACHÉ DE IA PARA VELOCIDAD ULTRA-RÁPIDA (< 1 ms) ──────────
@lru_cache(maxsize=1)
def _load_ai_model(mtime):
    try:
        model_path = config.AI_MODEL_FILE
        return joblib.load(model_path)
    except Exception:
        return None

def get_ai_model():
    model_path = config.AI_MODEL_FILE
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


def zscore_vwap(df: pd.DataFrame, window: int = 20):
    """
    Z-Score del precio respecto al VWAP — Reversión a la Media Institucional.

    Z-Score = (Precio - VWAP) / Desviación Estándar del precio

    Interpretación:
    - Z-Score > +2.0  → Sobrecompra estadística → NO comprar
    - Z-Score < -2.0  → Sobrevendido estadístico → Posible compra

    El VWAP como media de referencia es el estándar institucional porque
    considera dónde se negó el mayor volumen (equilibrio real del mercado).
    """
    if len(df) < window: return pd.Series(0, index=df.index)
    v = vwap(df)
    std = df["Close"].rolling(window).std()
    z = (df["Close"] - v) / std
    return z

# ─── Bandas de Bollinger ───────────────────────────────────────────────────
def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    if len(series) < window:
        return series, series, series
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

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


# ════════════════════════════════════════════════════════════════════════════
#  DETECTOR DE RÉGIMEN DE MERCADO (Market Regime Detection)
# ════════════════════════════════════════════════════════════════════════════

# 5 regímenes universales que cubren cualquier estado del mercado financiero
REGIMEN_LABELS = {
    "TREND_UP":    "🚀 Tendencia Alcista",
    "TREND_DOWN":  "📉 Tendencia Bajista",
    "RANGE":       "🎯 Mercado Lateral (Rango)",
    "MOMENTUM":    "⚡ Momentum Explosivo",
    "CHAOS":       "🌩️ Volatilidad Alta (Caos)",
    "NEUTRAL":     "☁️ Mercado Neutro",
}

def detect_regime(
    adx_val: float,
    atr_pct: float,
    close: float,
    ema_200: float,
    rsi_val: float,
    vol_ratio: float,
) -> str:
    """
    Clasifica el mercado en uno de 5 regímenes usando indicadores ya calculados.

    Parámetros:
        adx_val   -- Fuerza de la tendencia (ADX 14)
        atr_pct   -- ATR como % del precio (volatilidad normalizada)
        close     -- Precio de cierre actual
        ema_200   -- EMA 200 (tendencia macro)
        rsi_val   -- RSI actual (momentum)
        vol_ratio -- Volumen actual / promedio 20 barras

    Retórnos posibles:
        TREND_UP   -- Tendencia alcista clara. Estrategia: seguir la tendencia.
        TREND_DOWN -- Tendencia bajista clara. Estrategia: no comprar.
        RANGE      -- Mercado lateral estrecho. Estrategia: rebotes VWAP.
        MOMENTUM   -- Breakout explosivo. Estrategia: ride the wave con stop ajustado.
        CHAOS      -- Volatilidad extrema. Estrategia: no operar hasta que se calme.
        NEUTRAL    -- Condición mixta. Estrategia: criterios base.
    """
    # ── Caos: volatilidad extrema que destroza cualquier stop ────────────────
    if atr_pct > 1.5:
        return "CHAOS"

    # ── Momentum explosivo: volumen 3x + RSI alto pero no burbuja ────────────
    if vol_ratio >= 3.0 and 60 < rsi_val < 80 and close > ema_200:
        return "MOMENTUM"

    # ── Tendencias claras (ADX > 28 es umbral institucional real) ────────────
    if adx_val > 28:
        if close > ema_200:
            return "TREND_UP"
        else:
            return "TREND_DOWN"

    # ── Lateral: ADX bajo y volatilidad contenida ────────────────────────────
    if adx_val < 20 and atr_pct < 0.6:
        return "RANGE"

    # ── Resto: condición mixta, usar criterios base conservadores ───────────
    return "NEUTRAL"

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
        regime:        str  = "NEUTRAL",  # Régimen de mercado detectado
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
        self.regime        = regime

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

# Mínimo de barras para permitir inicio rápido con Alpaca Paper
MIN_BARS = 80  # Restaurado! Ahora usamos el cache compartido de 60 días (market_data.py)


def analyze(df: pd.DataFrame, symbol: str = "", asset_type: str = "normal") -> SignalResult:
    """
    Evalúa la señal de trading usando un sistema de confluencia multi-indicador.

    Requiere mínimo 3 confirmaciones alcistas y 0 bloqueos activos para emitir BUY.
    Esto reduce drásticamente las falsas señales del EMA crossover simple.

    Args:
        df: DataFrame OHLCV con suficientes barras históricas.
        symbol: Símbolo del activo (para identificación en logs).
        asset_type: Tipo de activo ("normal" o "inverted").

    Returns:
        SignalResult con señal final y todos los valores de indicadores.
    """
    # ── Diagnóstico de datos ──────────────────────────────────────────────────
    from shared.utils.logger import log as _log
    if symbol:
        _log.info(f"[{symbol}] Analizando {len(df)} barras históricas (requiere {MIN_BARS})")

    # ── Inicializar resultado neutro si no hay suficientes datos ─────────────
    if len(df) < MIN_BARS:
        return SignalResult(
            signal=SIGNAL_HOLD,
            ema_fast=0.0, ema_slow=0.0, ema_200=0.0,
            rsi_value=50.0, macd_hist=0.0,
            vwap_value=0.0, atr_value=0.0,
            close=float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0,
            timestamp=df.index[-1] if len(df) > 0 else None,
            confirmations=[], blocks=[f"⏳ Cargando datos ({len(df)}/{MIN_BARS})"],
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

    # ── Calcular vol_ratio para el detector de régimen ───────────────────────
    vol_series  = df["Volume"].astype(float)
    vol_avg     = float(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else float(vol_series.mean())
    vol_ratio   = float(vol_series.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
    atr_pct     = (atr_now / close_now) * 100 if close_now else 0.0
    
    # Pendiente del ADX (para detectar si la tendencia se está fortaleciendo)
    adx_prev    = float(adx_vals.iloc[-2]) if not pd.isna(adx_vals.iloc[-2]) else adx_now
    is_adx_rising = adx_now > adx_prev

    # ╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬
    #  DETECTOR DE RÉGIMEN ─ Se ejecuta PRIMERO
    # ╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬
    current_regime = detect_regime(
        adx_val   = adx_now,
        atr_pct   = atr_pct,
        close     = close_now,
        ema_200   = ema_200_now,
        rsi_val   = rsi_now,
        vol_ratio = vol_ratio,
    )
    regime_label = REGIMEN_LABELS.get(current_regime, current_regime)
    from shared.utils.logger import log as _log
    sym_prefix = f"[{symbol}] " if symbol else ""
    _log.info(f"{sym_prefix}🌍 RÉGIMEN DETECTADO: {regime_label} | ADX={adx_now:.1f} | ATR%={atr_pct:.2f}% | VolRatio={vol_ratio:.1f}x")

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
    vwap_oversold = zscore_now < -2.0
    vwap_extreme  = zscore_now < -3.0
    vwap_reversal = (close_now > open_now) or is_hammer
    macro_bullish = close_now > ema_200_now
    
    is_vwap_bounce = (vwap_oversold and vwap_reversal and macro_bullish) or (vwap_extreme and vwap_reversal)
    
    # ── ESTRATEGIA C: Reversión de Bandas de Bollinger ────────────────────
    # El precio perfora la banda inferior de Bollinger (Pánico) y luego 
    # recupera con un cierre superior al Low anterior o media de las bandas.
    bb_upper, bb_mid, bb_lower = bollinger_bands(closes)
    bb_low_now = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else 0.0
    
    is_bb_reversal = (low_now <= bb_low_now) and (close_now > open_now) # Touch + Green Close
    
    # ── Determinar si es un activo invertido (Inverse ETF) ────────────────────
    is_inverted = asset_type == "inverted"
    
    # ── EVALUACIÓN POR RÉGIMEN ═ El sistema elige la estrategia óptima ══
    
    # LÓGICA ESPECIAL PARA ETFs INVERSOS: El bot gana cuando el mercado general baja.
    # Por lo tanto, si el mercado está en TREND_DOWN, para el ETF inverso esto es una oportunidad.
    effective_regime = current_regime
    if is_inverted:
        if current_regime == "TREND_DOWN":
            effective_regime = "TREND_UP"     # Invertimos la percepción
        elif current_regime == "TREND_UP":
            effective_regime = "TREND_DOWN"   # Evitamos comprar si el mercado sube
            
    if effective_regime in ("TREND_UP", "MOMENTUM", "NEUTRAL"):
        if is_velez_setup:
            signal = SIGNAL_BUY
            prefix = "📉⚡ LÓGICA INVERSA | " if is_inverted else ""
            confirmations_buy.append(f"🎯 {prefix}[{regime_label}] Estrategia A: Rebote Oliver Vélez en la EMA 20.")

    if effective_regime in ("RANGE", "NEUTRAL"):
        if signal != SIGNAL_BUY and is_vwap_bounce:
            signal = SIGNAL_BUY
            prefix = "📉⚡ LÓGICA INVERSA | " if is_inverted else ""
            confirmations_buy.append(f"🎯 {prefix}[{regime_label}] Estrategia B: Anomalía VWAP (Z={zscore_now:.2f}) con rechazo alcista.")
        
        if signal != SIGNAL_BUY and is_bb_reversal:
            signal = SIGNAL_BUY
            prefix = "📉⚡ LÓGICA INVERSA | " if is_inverted else ""
            confirmations_buy.append(f"🎯 {prefix}[{regime_label}] Estrategia C: Reversión de Bandas de Bollinger (Toque Inferior).")

        # ── ESTRATEGIA D: Cruce de Medias Móviles (Clásica) ────────────────────
        ema_crossover = (ema_f_prev < ema_s_prev) and (ema_f_now > ema_s_now)
        # Filtro adicional: Solo si precio > EMA 200 y RSI saludable
        if signal != SIGNAL_BUY and ema_crossover and macro_bullish and (35 < rsi_now < 70):
            signal = SIGNAL_BUY
            prefix = "📉⚡ LÓGICA INVERSA | " if is_inverted else ""
            confirmations_buy.append(f"🎯 {prefix}[{regime_label}] Estrategia D: Cruce EMA Clásico ({config.EMA_FAST}/{config.EMA_SLOW}).")

    if effective_regime in ("TREND_DOWN", "CHAOS"):
        # AGRESIVO V3: Permitimos compras en tendencia bajista SOLO SI es un "Flash Crash"
        # detectado por un Z-Score extremo o una desviación gigante de BB.
        if (zscore_now < -2.5) and (is_hammer or is_engulf or is_bb_reversal):
            signal = SIGNAL_BUY
            prefix = "📉⚡ LÓGICA INVERSA | " if is_inverted else ""
            confirmations_buy.append(f"🔥 {prefix}[{regime_label}] COMPRA DE CAPITULACIÓN: Z-Score={zscore_now:.2f} con patrón de reversión.")
        else:
            signal = SIGNAL_HOLD
            confirmations_buy.clear()
        
    # ── FILTRO INTELIGENTE BASADO EN MACHINE LEARNING HISTÓRICO ───────────
    # AGRESIVO: Reducimos la zona de bloqueo del RSI (antes 40-50).
    is_whipsaw_zone = (rsi_now > 43.0) and (rsi_now < 47.0)
    
    # Filtro ADX: Evitamos laterales sin dirección.
    # AGRESIVO V3: Bajamos el umbral a 10.
    is_low_trend = adx_now < 10.0 and not (is_hammer or is_engulf or is_bb_reversal)
    
    if signal == SIGNAL_BUY and (is_whipsaw_zone or is_low_trend):
        # EXCEPCIÓN: Si es una compra de CAPITULACIÓN extrema, ignoramos el ADX (porque el ADX suele ser bajo en giros rápidos)
        if zscore_now < -2.3: 
            pass 
        else:
            signal = SIGNAL_HOLD
            if is_whipsaw_zone:
                blocks_buy.append(f"Filtro ML: RSI en zona muerta ({rsi_now:.1f})")
            if is_low_trend:
                blocks_buy.append(f"Filtro ADX: Tendencia débil ({adx_now:.1f} < 10)")
            from shared.utils.logger import log as _log
            _log.warning(f"{sym_prefix}🎯🛑 OPORTUNIDAD EVITADA: Momentum RSI/ADX insuficiente.")
            confirmations_buy.clear()

    # ── REGLA TREND_UP: Eliminamos el bloqueo de RSI > 68 para permitir momentum ──
    # En una tendencia fuerte alcista, el RSI puede mantenerse alto mucho tiempo.
    pass 

    # ── NUEVA REGLA: RANGE + Z-Score débil → señal falsa en mercado lateral ──
    # AGRESIVO: Excepción Pro: Si el Z-Score es suficientemente bajo (< -1.8), permitimos la entrada.
    if signal == SIGNAL_BUY and current_regime == "RANGE" and zscore_now > -1.8:
        signal = SIGNAL_HOLD
        msg = f"RANGE: Z-Score insuficiente ({zscore_now:.2f} > -1.8)"
        blocks_buy.append(msg)
        confirmations_buy.clear()
        from shared.utils.logger import log as _log
        _log.warning(f"{sym_prefix}🎯� OPORTUNIDAD EVITADA: {msg}")

    if signal == SIGNAL_BUY and (20.0 < adx_now < 30.0) and is_adx_rising:
        if current_regime in ("NEUTRAL", "RANGE"):
            pass
            if not (is_hammer or is_engulf):
                signal = SIGNAL_HOLD
                msg = f"SMART FILTER: ADX en zona de duda ({adx_now:.1f}) y sin patrón de vela."
                blocks_buy.append(msg)
                confirmations_buy.clear()
                from shared.utils.logger import log as _log
                _log.warning(f"{sym_prefix}🎯� OPORTUNIDAD EVITADA: {msg}")
        
    # ── LÓGICA DE VENTA DE EMERGENCIA ─────────────────────────────────────
    # Tu Take Profit (3%) o Trailing Stop harán el 90% del trabajo de salida. 
    # Pero forzamos venta de pánico solo si perfora severamente la EMA 20 + MACD se torna bajista agresivo
    force_sell = (close_now < ema_20_now) and (macd_now < macd_prev) and (macd_now < 0)
    
    if force_sell and signal == SIGNAL_HOLD:
        signal = SIGNAL_SELL
        reason = "Estructura deteriorada (perdió EMA 20 + MACD bajista)"
        confirmations_sell.append(f"🛡️ Venta forzada: {reason}")
        blocks_buy.append(f"CRÍTICO: {reason}")
        
    if signal == SIGNAL_HOLD and not blocks_buy:
        if current_regime == "TREND_DOWN":
            blocks_buy.append("Bloqueo: Tendencia bajista macro")
        elif current_regime == "CHAOS":
            blocks_buy.append("Riesgo: Volatilidad extrema (Caos)")
        else:
            blocks_buy.append("Sin señal clara (Wait)")

    # ── GENERACIÓN DE FEATURES PARA MACHINE LEARNING ──────────────────────────
    ml_features = {
        'rsi': float(rsi_now),
        'macd_hist': float(macd_now),
        'ema_diff_pct': float((ema_f_now - ema_s_now) / close_now * 100) if close_now else 0,
        'vwap_dist_pct': float((close_now - vwap_now) / vwap_now * 100) if vwap_now > 0 else 0,
        'atr_pct': float(atr_pct),
        'adx': float(adx_now),
        'regime': current_regime,
        'vol_ratio': float(vol_ratio),
        'ema_fast': float(ema_f_now),
        'ema_slow': float(ema_s_now),
        'zscore_vwap': float(zscore_now),
        'num_confirmations': len(confirmations_buy),
    }

    # 🤖 EJECUCIÓN DEL MODELO DE INTELIGENCIA ARTIFICIAL EN VIVO
    # Consulta al oráculo del Machine Learning antes de abrir fuego.
    if signal == SIGNAL_BUY:
        # 1. Modelo RandomForest heredado (si existe)
        ai_model = get_ai_model()
        if ai_model:
            try:
                expected_cols = ai_model.feature_names_in_
                X_live = pd.DataFrame([ml_features])[expected_cols]
                if ai_model.predict(X_live)[0] == 0:
                    signal = SIGNAL_HOLD
                    confirmations_buy.clear()
            except Exception as e:
                from shared.utils.logger import log
                log.error(f"Error AI prediction: {e}")

    # 2. Red Neuronal MLP (filtro de pre-aprobación adaptativo)
    if signal == SIGNAL_BUY:
        try:
            from shared.utils.neural_filter import get_neural_filter
            nf = get_neural_filter()
            from shared.config import CONFIDENCE_THRESHOLD as _MIN_CONF
            
            features_vec = nf.build_features(
                rsi          = rsi_now,
                macd_hist    = macd_now,
                atr_pct      = atr_pct,
                vol_ratio    = vol_ratio,
                ema_fast     = ema_f_now,
                ema_slow     = ema_s_now,
                zscore_vwap  = zscore_now,
                regime       = current_regime,
                num_confirmations = len(confirmations_buy),
            )
            proba, reason = nf.predict(features_vec)

            from shared.utils.logger import log as _log
            _log.info(f"🧠 Red Neuronal: {reason}")

            if proba < _MIN_CONF:   
                signal = SIGNAL_HOLD
                confirmations_buy.clear()
                blocks_buy.append(f"🧠 Red Neuronal bloqueó la entrada: {reason}")
            else:
                confirmations_buy.append(f"🧠 Red Neuronal: aprobado (P={proba:.0%})")

        except Exception as e:
            from shared.utils.logger import log as _log
            _log.warning(f"Error en filtro neuronal, omitiendo: {e}")


    # Logeamos la ejecución del escuadrón (con RÉGIMEN)
    if signal == SIGNAL_BUY:
        from shared.utils.logger import log
        log.info(f"🔫 COMPRA PERMITIDA | [{regime_label}] | [ADX={adx_now:.1f}] | " + " | ".join(confirmations_buy))
    elif signal == SIGNAL_SELL:
        from shared.utils.logger import log
        _log.info(f"{sym_prefix}🔴 VENTA ALERTA TÁCTICA | [{regime_label}] | [ADX={adx_now:.1f}] | " + " | ".join(confirmations_sell))

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
        ml_features=ml_features,
        regime=current_regime,
    )
