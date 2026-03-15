"""
strategy/patterns.py ─ Detección de patrones de velas japonesas.

Responsabilidad única: proveer funciones para detectar martillos, engulfing
y picos de volumen.
"""

import pandas as pd

def volume_spike(df: pd.DataFrame, window: int = 20, threshold: float = 1.5) -> pd.Series:
    """Detecta si el volumen actual supera significativamente el promedio móvil."""
    vol     = df["Volume"].astype(float)
    avg_vol = vol.rolling(window).mean()
    return vol > (avg_vol * threshold)

def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detecta el patrón Hammer (martillo) alcista."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    return (lower_wick >= 2 * body.replace(0, 0.0001)) & (upper_wick <= body * 0.5) & (body > 0)

def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detecta el patrón Engulfing alcista."""
    o, c = df["Open"], df["Close"]
    po, pc = o.shift(1), c.shift(1)
    return (pc < po) & (c > o) & (c > po) & (o < pc)
