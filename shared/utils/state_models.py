"""
utils/state_models.py ─ Modelos de datos para el estado del bot y registros de trades.

Contiene las clases TradeRecord y BotState que definen la estructura del
estado que se envía al dashboard.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class TradeRecord:
    symbol: str
    side:  str
    price: float
    qty:   float
    pnl:   float
    time:  str = ""
    reason: str = ""
    confirmations: list = field(default_factory=list)
    ml_prob: float = 0.0
    conf_mult: float = 1.0
    entry_reason: str = ""
    entry_price: float = 0.0
    date: str = ""
    fees: float = 0.0
    xai_metrics: dict = field(default_factory=dict)

@dataclass
class BotState:
    # Identificación
    mode:    str = "SIMULATED"
    symbol:  str = "─"
    session: int = 1
    iteration: int = 0
    timestamp: str = ""

    # Precios
    bid: float = 0.0
    ask: float = 0.0

    # Señal
    signal:   str   = "HOLD"
    regime:   str   = "NEUTRAL"
    rsi:      float = 50.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_200:  float = 0.0
    macd_hist: float = 0.0
    vwap:     float = 0.0
    atr:      float = 0.0
    confirmations: list = field(default_factory=list)

    # Cuenta
    initial_cash:   float = 10_000.0
    available_cash: float = 10_000.0
    settlement:     float = 0.0

    # Estadísticas
    win_rate:       float = 0.0
    total_trades:   int   = 0
    winning_trades: int   = 0
    gross_profit:   float = 0.0
    gross_loss:     float = 0.0
    total_fees:     float = 0.0
    total_slippage: float = 0.0
    
    # Estadísticas Acumuladas
    total_sim_trades: int = 0
    total_sim_wins:   int = 0
    total_sim_pnl:    float = 0.0
    total_sim_ghosts: int = 0
    total_sim_fees:   float = 0.0
    total_sim_slippage: float = 0.0
    total_sim_gross_profit: float = 0.0
    total_sim_gross_loss:   float = 0.0
    
    # 👻 Fantasmas (Activo)
    total_ghosts:       int = 0
    ghost_trades_count: int = 0

    # Posición abierta
    position: Optional[dict] = None
    trades: list = field(default_factory=list)
    candles: list = field(default_factory=list)

    status: str = "running"
    next_scan_in: int = 0
    is_waiting: bool = False
    mock_time_930: bool = False
    blocks: List[str] = field(default_factory=list)
    blocking_summary: dict = field(default_factory=dict)
    sim_start: str = ""
    sim_end:   str = ""
    sim_duration: float = 0.0
    mock_time: str = ""  # Reloj simulado para modo Replay

    mock_time: str = ""  # Reloj simulado para modo Replay


    # 🧠 Información de la IA Asesora
    ai_win_prob: float = 0.0
    ai_recommendation: str = ""
    ai_expected_up: float = 0.0
    ai_expected_down: float = 0.0
    model_accuracy: float = 0.0
    total_samples:  int = 0
    is_ml_blocked: bool = False
    is_quality_blocked: bool = False
    last_action: str = ""

    # 🏆 Certificación de Maestría y Gestión de Capital
    mastery_score: float = 0.0          # 0-100%
    mastery_status: str = "APRENDIENDO" # APRENDIENDO, VALIDANDO, MAESTRÍA
    recommended_cash: float = 500.0     # Capital sugerido para el riesgo actual
    risk_tier: str = "Normal (1x)"      # Normal, Agresivo, Alto Riesgo
    actual_profit_factor: float = 0.0
    actual_max_drawdown: float = 0.0
    mastery_checklist: List[str] = field(default_factory=list)

    # --- Analyst Upgrade ---
    expectancy: float = 0.0
    efficiency: float = 0.0
    stability_score: float = 0.0
    equity_history: List[float] = field(default_factory=list)

