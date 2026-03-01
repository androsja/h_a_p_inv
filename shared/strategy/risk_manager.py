"""
strategy/risk_manager.py ─ Gestión de riesgo profesional (nivel institucional).

Reglas implementadas:
  1.  Stop-Loss DINÁMICO basado en ATR (se adapta a la volatilidad real).
  2.  Take-Profit 3.0%: ratio Riesgo:Recompensa mínimo 1:2.
  3.  TRAILING STOP: el SL sube automáticamente a medida que el precio sube.
  4.  Cierre por TIEMPO: si la posición no se mueve en 20 min → cerrar.
  5.  Tamaño de posición limitado por MAX_POSITION_USD.
  6.  La operación DEBE superar el costo de la cámara Apex ($0.15).
  7.  Verificación de buying_power (descontando cash en T+1 settlement).
  8.  Ajuste de precio límite por latencia geográfica Colombia→NY.
"""

from dataclasses import dataclass, field
from datetime    import date, datetime, timezone
from shared import config
from shared.utils.logger import log, log_risk_block, log_settlement_block


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
    ml_features:   dict  = field(default_factory=dict) # Features al momento de abrir la posición
    hold_bars:     int   = 0    # Iteraciones que lleva abierta la posición
    entry_reason:  str   = ""   # Razón de la entrada
    entry_metadata: dict = field(default_factory=dict) # Metadata de la entrada

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
        """Minutos que lleva abierta la posición."""
        delta = datetime.now(timezone.utc) - self.opened_at
        return delta.total_seconds() / 60


# ════════════════════════════════════════════════════════════════════════════
#  ESTADO DE LA CUENTA
# ════════════════════════════════════════════════════════════════════════════

class AccountState:
    """
    Rastrea el capital disponible y el cash pendiente por settlement T+1.
    En Hapi (Apex Clearing), el dinero de una venta no está disponible
    para comprar el mismo día — se liquida al día hábil siguiente.
    """

    def __init__(self, total_cash: float):
        self.total_cash: float = total_cash
        self._pending_settlement: dict[date, float] = {}
        self.open_position: OpenPosition | None = None

    def record_sale(self, amount: float) -> None:
        """Registra capital de una venta como bloqueado hasta T+1."""
        today = date.today()
        self._pending_settlement[today] = (
            self._pending_settlement.get(today, 0.0) + amount
        )
        log.info(
            f"settlement | ${amount:.2f} bloqueados en T+1 "
            f"(disponibles mañana, {today})"
        )

    def release_settled_cash(self) -> None:
        """Libera capital de ventas de días anteriores."""
        today     = date.today()
        released  = {d: a for d, a in self._pending_settlement.items() if d < today}
        for d, amt in released.items():
            self.total_cash += amt
            del self._pending_settlement[d]
            log.info(f"settlement | ${amt:.2f} del {d} liberados.")

    @property
    def available_cash(self) -> float:
        self.release_settled_cash()
        blocked = sum(self._pending_settlement.values())
        return max(0.0, self.total_cash - blocked)

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
    atr_stop:     float = 0.0   # SL calculado con ATR


# ════════════════════════════════════════════════════════════════════════════
#  RISK MANAGER
# ════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Calcula el tamaño de posición, niveles de SL/TP y evalúa la salida.

    Mejoras profesionales vs versión anterior:
      • SL dinámico: max(SL_fijo%, SL_ATR)  → se adapta a la volatilidad
      • Trailing Stop: sube el SL cuando el precio sube
      • Cierre por tiempo: 20 minutos sin movimiento → cerrar
    """

    # Tiempo máximo en una posición sin alcanzar SL o TP (minutos)
    MAX_MINUTES_IN_POSITION: int = 390  # Relajamos el pánico: le damos toda la jornada (6.5 horas) para madurar la tendencia

    def __init__(self, account: AccountState):
        self.account = account

    # ── CÁLCULO DE ORDEN DE COMPRA ───────────────────────────────────────────
    def calculate_buy_order(
        self,
        symbol:    str,
        ask_price: float,
        atr_value: float = 0.0,
        confidence_multiplier: float = 1.0,
    ) -> OrderPlan:
        """
        Calcula parámetros de BUY con SL dinámico basado en ATR.

        Stop Loss = max(SL_fijo%, 1.5 × ATR)
        Take Profit siempre = 2 × Stop Loss (ratio 1:2 garantizado)
        """
        limit_price = round(ask_price + config.LATENCY_OFFSET_CENTS, 4)

        if limit_price <= 0:
            return OrderPlan(
                symbol=symbol, side="BUY", limit_price=limit_price,
                qty=0, stop_loss=0, take_profit=0,
                min_profit=config.CLEARING_COST_USD,
                is_viable=False,
                block_reason="Precio inválido",
            )

        # ── Stop Loss dinámico (ATR-based) ──────────────────────────────────
        # SL fijo: porcentaje configurado (1.5%)
        sl_fixed = limit_price * config.STOP_LOSS_PCT

        # SL basado en ATR: 1.5 × ATR (se adapta a la volatilidad del momento)
        # Si ATR = $2.50, el SL es $3.75 debajo del precio de entrada
        sl_atr = (atr_value * 1.5) if atr_value > 0 else sl_fixed

        # Usamos el mayor de los dos: protección mínima garantizada
        sl_distance = max(sl_fixed, sl_atr)
        stop_loss   = round(limit_price - sl_distance, 4)

        # ── Tamaño de posición basado en RIESGO (Kelly Fraccionado) ────────
        available    = self.account.available_cash
        
        # Riesgo máximo que queremos asumir en $ (por default 1% de la cuenta)
        max_risk_usd = available * getattr(config, 'MAX_RISK_PCT', 0.01)
        
        # Si perdemos exactamente sl_distance por acción, ¿cuántas podemos comprar sin superar max_risk_usd?
        qty_by_risk = max_risk_usd / sl_distance if sl_distance > 0 else 0
        
        # Nunca comprar más del capital total ni de la configuración máxima permitida
        max_notional = min(available, config.MAX_POSITION_USD)
        qty_by_cash  = max_notional / limit_price if limit_price > 0 else 0
        
        # El tamaño final es el MÍNIMO entre lo que nos dicta el Riesgo, y lo que nos da el Cash en el bolsillo
        # APLICAMOS EL FACTOR DE CONFIANZA: Si es una inversión segura, aumentamos un poco el tamaño
        base_qty = min(qty_by_risk, qty_by_cash)
        qty = round(base_qty * confidence_multiplier, 4)

        if qty <= 0:
            return OrderPlan(
                symbol=symbol, side="BUY", limit_price=limit_price,
                qty=0, stop_loss=0, take_profit=0,
                min_profit=config.CLEARING_COST_USD,
                is_viable=False,
                block_reason="Sin capital o restricción de riesgo le impide operar este símbolo hoy.",
            )

        # ── Take Profit siempre 2:1 vs el SL real ───────────────────────────
        # Si el SL es $3.75, el TP debe ser al menos $7.50 de ganancia
        tp_distance = sl_distance * 2          # ratio R:R = 1:2
        take_profit = round(limit_price + tp_distance, 4)

        # ── Verificación de rentabilidad ────────────────────────────────────
        gross_profit = tp_distance * qty
        if gross_profit < config.CLEARING_COST_USD:
            reason = (
                f"Ganancia esperada ${gross_profit:.4f} < "
                f"costo Apex ${config.CLEARING_COST_USD}"
            )
            log_risk_block(symbol, reason)
            return OrderPlan(
                symbol=symbol, side="BUY", limit_price=limit_price,
                qty=qty, stop_loss=stop_loss, take_profit=take_profit,
                min_profit=config.CLEARING_COST_USD,
                is_viable=False, block_reason=reason, atr_stop=sl_atr,
            )

        log.debug(
            f"RiskMgr | {symbol} | "
            f"SL_fijo=${sl_fixed:.2f} SL_ATR=${sl_atr:.2f} → "
            f"SL final=${stop_loss:.2f} TP=${take_profit:.2f}"
        )

        return OrderPlan(
            symbol=symbol, side="BUY", limit_price=limit_price,
            qty=qty, stop_loss=stop_loss, take_profit=take_profit,
            min_profit=config.CLEARING_COST_USD,
            is_viable=True, atr_stop=sl_atr,
        )

    # ── CÁLCULO DE ORDEN DE VENTA ────────────────────────────────────────────
    def calculate_sell_order(self, position: OpenPosition, bid_price: float) -> OrderPlan:
        """Orden de venta con ajuste de latencia."""
        limit_price = round(bid_price - config.LATENCY_OFFSET_CENTS, 4)
        return OrderPlan(
            symbol=position.symbol, side="SELL", limit_price=limit_price,
            qty=position.qty, stop_loss=0, take_profit=0,
            min_profit=config.CLEARING_COST_USD,
            is_viable=True,
        )

    # ── EVALUACIÓN DE SALIDA ─────────────────────────────────────────────────
    def should_exit(
        self,
        position:      OpenPosition,
        current_price: float,
    ) -> tuple[bool, str]:
        """
        Evalúa si la posición actual debe cerrarse.
        Verifica: SL | TP | TRAILING STOP | CIERRE POR TIEMPO
        Returns: (debe_cerrar, motivo)
        """

        # ── 1. TRAILING STOP ────────────────────────────────────────────────
        # Si el precio subió y luego retrocedió, proteger ganancias
        position = self._update_trailing_stop(position, current_price)

        # ── 2. Stop Loss dinámico (incluye trailing) ─────────────────────────
        if current_price <= position.stop_loss:
            pnl = position.pnl(current_price)
            is_trailing = position.stop_loss > position.initial_stop
            label = "🔒 TRAILING_STOP" if is_trailing else "🛑 STOP_LOSS"
            return True, (
                f"{label} alcanzado "
                f"(${current_price:.2f} ≤ ${position.stop_loss:.2f}) "
                f"PnL={pnl:+.2f}"
            )

        # ── 3. Take Profit ───────────────────────────────────────────────────
        if current_price >= position.take_profit:
            pnl = position.pnl(current_price)
            return True, (
                f"🎯 TAKE_PROFIT alcanzado "
                f"(${current_price:.2f} ≥ ${position.take_profit:.2f}) "
                f"PnL={pnl:+.2f}"
            )

        # ── 4. Cierre por tiempo ─────────────────────────────────────────────
        minutes = position.minutes_open
        if minutes >= self.MAX_MINUTES_IN_POSITION:
            pnl = position.pnl(current_price)
            return True, (
                f"⏰ CIERRE_POR_TIEMPO "
                f"({minutes:.0f} min ≥ {self.MAX_MINUTES_IN_POSITION} min) "
                f"PnL={pnl:+.2f} — capital liberado para otra oportunidad"
            )

        return False, ""

    # ── TRAILING STOP INTERNO ────────────────────────────────────────────────
    def _update_trailing_stop(
        self, position: OpenPosition, current_price: float
    ) -> OpenPosition:
        """
        Mueve el Stop Loss hacia arriba cuando el precio sube.

        Regla: el trailing stop se activa cuando la posición está en ganancia
        de al menos 1%. A partir de ahí, el SL sigue al precio manteniendo
        la misma distancia que el SL original.
        """
        # Solo activar el trailing si el precio ya está 1% arriba
        activation_pct = 0.01
        if current_price < position.entry_price * (1 + activation_pct):
            return position

        # Actualizar el precio máximo alcanzado
        if current_price > position.highest_price:
            position.highest_price = current_price

            # Calcular nuevo SL: precio máximo - distancia original del SL
            original_sl_distance = position.entry_price - position.initial_stop
            new_sl = round(position.highest_price - original_sl_distance, 4)

            # Solo subir el SL, nunca bajarlo
            if new_sl > position.stop_loss:
                old_sl = position.stop_loss
                position.stop_loss = new_sl
                log.debug(
                    f"🔒 TRAILING_STOP | {position.symbol} | "
                    f"Nuevo máximo=${current_price:.2f} → "
                    f"SL: ${old_sl:.2f} → ${new_sl:.2f}"
                )

        return position
