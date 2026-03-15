"""
strategy/signal_models.py ─ Modelo de datos para el resultado del análisis.
"""

from dataclasses import dataclass, field

@dataclass
class SignalResult:
    """Resultado completo del análisis técnico multi-indicador."""
    signal:        str
    ema_fast:      float
    ema_slow:      float
    ema_200:       float
    rsi_value:     float
    macd_hist:     float
    vwap_value:    float
    atr_value:     float
    close:         float
    timestamp:     any
    confirmations: list[str] = field(default_factory=list)
    blocks:        list[str] = field(default_factory=list)
    ml_features:   dict = field(default_factory=dict)
    regime:        str = "NEUTRAL"

    def __repr__(self) -> str:
        return (
            f"SignalResult(signal={self.signal} | "
            f"✅ {len(self.confirmations)} confirman | "
            f"❌ {len(self.blocks)} bloquean | "
            f"RSI={self.rsi_value:.1f} MACD={self.macd_hist:.3f})"
        )
