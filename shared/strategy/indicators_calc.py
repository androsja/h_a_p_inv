"""
strategy/indicators_calc.py ─ Cálculos matemáticos de indicadores técnicos.

Responsabilidad única: proveer funciones puras para el cálculo de
indicadores (EMA, RSI, ADX, MACD, ATR, VWAP, KAMA, etc).
"""

import math
import numpy as np
import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) clásico de Wilder."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX)."""
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    close = df['Close'].astype(float)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    pos_dm_smooth = pd.Series(pos_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    neg_dm_smooth = pd.Series(neg_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    
    di_plus = 100 * (pos_dm_smooth / tr_smooth.replace(0, np.nan))
    di_minus = 100 * (neg_dm_smooth / tr_smooth.replace(0, np.nan))
    
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD Line, Signal Line, Histogram."""
    ema_f   = series.ewm(span=fast, adjust=False).mean()
    ema_s   = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP (Volume Weighted Average Price) acumulativo."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].astype(float).replace(0, 1)
    return (tp * vol).cumsum() / vol.cumsum()

def zscore_vwap(df: pd.DataFrame, window: int = 20):
    """Z-Score del precio respecto al VWAP."""
    if len(df) < window: return pd.Series(0, index=df.index)
    v = vwap(df)
    std = df["Close"].rolling(window).std()
    return (df["Close"] - v) / std

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Upper, Mid, Lower Bollinger Bands."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return sma + (std * num_std), sma, sma - (std * num_std)

def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    f_sc = 2 / (fast + 1)
    s_sc = 2 / (slow + 1)
    prices = series.astype(float).values
    n = len(prices)
    k = np.full(n, np.nan)
    if n <= period: return series
    k[period] = prices[period]
    for i in range(period + 1, n):
        direction = abs(prices[i] - prices[i - period])
        vol = sum(abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1))
        er = direction / vol if vol > 0 else 0.0
        sc = (er * (f_sc - s_sc) + s_sc) ** 2
        k[i] = k[i - 1] + sc * (prices[i] - k[i - 1])
    return pd.Series(k, index=series.index)

def super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """Filtro Super Smoother de John Ehlers."""
    a1 = math.exp(-math.sqrt(2) * math.pi / period)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / period)
    c2, c3, c1 = b1, -(a1 ** 2), (1 - b1 + a1 ** 2) / 4
    p = series.astype(float).values
    n = len(p)
    ss = np.zeros(n)
    for i in range(2, n):
        ss[i] = c1 * (p[i] + p[i - 1]) + c2 * ss[i - 1] + c3 * ss[i - 2]
    r = pd.Series(ss, index=series.index)
    r[:2] = np.nan
    return r
