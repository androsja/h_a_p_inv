"""
strategy/regimes.py ─ Clasificación del régimen de mercado.

Diferencia mercados alcistas, bajistas, laterales, explosivos o caóticos
para adaptar la estrategia de trading.
"""

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
    """Clasifica el mercado en uno de los 5 regímenes universales."""
    if atr_pct > 1.5: return "CHAOS"
    if vol_ratio >= 3.0 and 60 < rsi_val < 80 and close > ema_200: return "MOMENTUM"
    if adx_val > 28:
        return "TREND_UP" if close > ema_200 else "TREND_DOWN"
    if adx_val < 20 and atr_pct < 0.6: return "RANGE"
    return "NEUTRAL"
