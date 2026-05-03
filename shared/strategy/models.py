"""
strategy/models.py ─ Modelos de datos (DTOs) del dominio de riesgo y posiciones.

Contiene las estructuras de datos puras sin lógica de negocio.
Son utilizadas tanto por RiskManager como por el bucle principal del bot.
"""

from dataclasses import dataclass, field
from datetime    import date, datetime, timezone
from typing import Optional, Union, Dict, List

from shared.utils.logger import log


# ════════════════════════════════════════════════════════════════════════════
#  POSICIÓN ABIERTA
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OpenPosition:
    symbol:       str
    entry_price:  float
    qty:          float
    stop_loss:    float          # SL dinámico (se mueve con trailing stop)
    take_profit:  float
    order_date:   date    = field(default_factory=date.today)
    opened_at:    datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    highest_price: float = 0.0  # Máximo precio alcanzado (para trailing stop)
    initial_stop:  float = 0.0  # SL original (para referencia)
    ml_features:   dict  = field(default_factory=dict)
    hold_bars:     int   = 0    # Iteraciones que lleva abierta la posición
    is_ghost:      bool  = False # 👻 Si es True, no afecta balance real ni se envía al bróker
    entry_reason:  str   = ""
    entry_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.initial_stop == 0.0:
            self.initial_stop = self.stop_loss

    @property
    def notional(self) -> float:
        return self.entry_price * self.qty

    def pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.qty

    @property
    def minutes_open(self) -> float:
        """Minutos que lleva abierta la posición en base al flujo de datos.
        Asegura un comportamiento idéntico en Simulación y en Vivo."""
        # Suponiendo velas de 5 minutos según config.DATA_INTERVAL
        return float(self.hold_bars * 5.0)


# ════════════════════════════════════════════════════════════════════════════
#  ESTADO DE LA CUENTA
# ════════════════════════════════════════════════════════════════════════════

class AccountState:
    """
    Rastrea el capital disponible y el cash pendiente por settlement T+1.
    En Hapi (IBKR Clearing), el dinero de una venta no está disponible
    para comprar el mismo día — se liquida al día hábil siguiente.
    """

    def __init__(self, total_cash: float):
        self.total_cash: float = total_cash
        self._pending_settlement: dict[date, float] = {}
        self.open_position: Optional[OpenPosition] = None
        self.current_market_date: date = date.today()  # Actualizado dinámicamente por el engine

    def record_sale(self, amount: float) -> None:
        """Registra capital de una venta como bloqueado hasta T+1."""
        today = self.current_market_date
        self._pending_settlement[today] = (
            self._pending_settlement.get(today, 0.0) + amount
        )
        log.info(
            f"settlement | ${amount:.2f} bloqueados en T+1 "
            f"(disponibles mañana, {today})"
        )

    def release_settled_cash(self) -> None:
        """Libera capital de ventas de días anteriores."""
        today    = self.current_market_date
        released = {d: a for d, a in self._pending_settlement.items() if d < today}
        for d, amt in released.items():
            self.total_cash += amt
            del self._pending_settlement[d]
            log.info(f"settlement | ${amt:.2f} del {d} liberados.")

    @property
    def available_cash(self) -> float:
        self.release_settled_cash()
        blocked = sum(self._pending_settlement.values())
        cash = self.total_cash - blocked
        if self.open_position:
            cash -= (self.open_position.qty * self.open_position.entry_price)
        return max(0.0, cash)

    @property
    def pending_settlement_total(self) -> float:
        return sum(self._pending_settlement.values())

    def summary(self) -> str:
        return (
            f"Cash total=${self.total_cash:.2f} | "
            f"Disponible=${self.available_cash:.2f} | "
            f"En T+1=${self.pending_settlement_total:.2f}"
        )


# ════════════════════════════════════════════════════════════════════════════
#  PLAN DE ORDEN
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OrderPlan:
    """Plan calculado por RiskManager antes de enviarlo al bróker."""
    symbol:       str
    side:         str     # "BUY" | "SELL"
    limit_price:  float
    qty:          float
    stop_loss:    float
    take_profit:  float
    min_profit:   float
    is_viable:    bool
    block_reason: str = ""
    atr_stop:     float = 0.0
