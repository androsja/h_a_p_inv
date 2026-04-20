"""
strategy/risk_manager.py ─ Gestión de riesgo profesional (nivel institucional).

Responsabilidad única: calcular tamaños de posición, niveles SL/TP y
evaluar condiciones de salida. Los modelos de datos (DTOs) están en models.py.

Reglas implementadas:
  1.  Stop-Loss DINÁMICO basado en ATR (se adapta a la volatilidad real).
  2.  Take-Profit 3.0%: ratio Riesgo:Recompensa mínimo 1:2.
  3.  TRAILING STOP: el SL sube automáticamente a medida que el precio sube.
  4.  Cierre por TIEMPO: si la posición no se mueve en 20 min → cerrar.
  5.  Tamaño de posición limitado por MAX_POSITION_USD.
  6.  La operación DEBE superar el costo de la cámara IBKR ($0.15).
  7.  Verificación de buying_power (descontando cash en T+1 settlement).
  8.  Ajuste de precio límite por latencia geográfica Colombia→NY.
"""

import json
from shared import config
from shared.config import COMMAND_FILE
from shared.utils.logger import log, log_risk_block

# Importar DTOs desde su propio módulo
from shared.strategy.models import OpenPosition, AccountState, OrderPlan

# Re-exportar para que los módulos que hacen `from shared.strategy.risk_manager import OpenPosition`
# sigan funcionando sin cambios (compatibilidad hacia atrás).
__all__ = ["RiskManager", "OpenPosition", "AccountState", "OrderPlan"]


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
    MAX_MINUTES_IN_POSITION: int = 390  # Toda la jornada (6.5 horas)

    def __init__(self, account: AccountState):
        self.account = account

    # ── MULTIPLICADOR DE INVERSIÓN ────────────────────────────────────────
    @staticmethod
    def _read_user_multiplier() -> float:
        """Lee el trade_multiplier desde command.json. Default = 1.0."""
        try:
            if COMMAND_FILE.exists():
                with open(COMMAND_FILE, "r") as cf:
                    cdata = json.load(cf)
                    return float(cdata.get("trade_multiplier", 1.0))
        except Exception:
            pass
        return 1.0

    # ── CÁLCULO DE ORDEN DE COMPRA ─────────────────────────────────────────
    def calculate_buy_order(
        self,
        symbol:    str,
        ask_price: float,
        atr_value: float = 0.0,
        swing_low: float = 0.0,
        confidence_multiplier: float = 1.0,
    ) -> OrderPlan:
        """
        Calcula parámetros de BUY con SL dinámico avanzado.

        Stop Loss = max(SL_fijo%, 1.5 × ATR, Swing_Low - buffer)
        Take Profit siempre = 2 × Stop Loss (ratio 1:2 garantizado)
        """
        limit_price = round(ask_price + config.LATENCY_OFFSET_CENTS, 4)

        if limit_price <= 0:
            return OrderPlan(
                symbol=symbol, side="BUY", limit_price=limit_price,
                qty=0, stop_loss=0, take_profit=0,
                min_profit=config.TARGET_MIN_NET_PROFIT_USD,
                is_viable=False, block_reason="Precio inválido",
            )

        # ── Stop Loss dinámico (ATR + Swing Low based) ──────────────────────
        sl_fixed = limit_price * config.STOP_LOSS_PCT
        sl_atr   = (atr_value * 1.5) if atr_value > 0 else sl_fixed
        
        # Swing Low: Usamos el mínimo reciente con un pequeño respiro (buffer)
        # El buffer evita que el SL esté "pegado" al mínimo exacto
        buffer = atr_value * 0.2 if atr_value > 0 else (limit_price * 0.001)
        sl_swing = (limit_price - (swing_low - buffer)) if swing_low > 0 else 0
        
        sl_distance = max(sl_fixed, sl_atr, sl_swing)
        stop_loss   = round(limit_price - sl_distance, 4)

        # ── Tamaño de posición basado en RIESGO (Kelly Fraccionado) ────────
        available    = self.account.available_cash
        max_risk_usd = available * getattr(config, 'MAX_RISK_PCT', 0.01)
        qty_by_risk  = max_risk_usd / sl_distance if sl_distance > 0 else 0

        max_notional = min(available, config.MAX_POSITION_USD)
        qty_by_cash  = max_notional / limit_price if limit_price > 0 else 0
        base_qty     = min(qty_by_risk, qty_by_cash)

        # ── Multiplicador del usuario (UI → command.json) ──────────────────
        user_multiplier = self._read_user_multiplier()
        
        # Seguridad crítica: no se puede superar lo que cubre el efectivo disponible
        qty = round(min(base_qty * user_multiplier, qty_by_cash) * confidence_multiplier, 4)

        if qty <= 0:
            return OrderPlan(
                symbol=symbol, side="BUY", limit_price=limit_price,
                qty=0, stop_loss=0, take_profit=0,
                min_profit=config.TARGET_MIN_NET_PROFIT_USD,
                is_viable=False,
                block_reason="Sin capital o restricción de riesgo impide operar hoy.",
            )

        # ── Take Profit siempre 2:1 vs el SL real ──────────────────────────
        tp_distance = sl_distance * 2
        take_profit = round(limit_price + tp_distance, 4)

        # ── Verificación de rentabilidad ────────────────────────────────────
        gross_profit = tp_distance * qty
        if gross_profit < config.TARGET_MIN_NET_PROFIT_USD:
            reason = (
                f"Ganancia esperada ${gross_profit:.4f} < "
                f"costo IBKR ${config.TARGET_MIN_NET_PROFIT_USD}"
            )
            log_risk_block(symbol, reason)
            return OrderPlan(
                symbol=symbol, side="BUY", limit_price=limit_price,
                qty=qty, stop_loss=stop_loss, take_profit=take_profit,
                min_profit=config.TARGET_MIN_NET_PROFIT_USD,
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
            min_profit=config.TARGET_MIN_NET_PROFIT_USD,
            is_viable=True, atr_stop=sl_atr,
        )

    # ── CÁLCULO DE ORDEN DE VENTA ──────────────────────────────────────────
    def calculate_sell_order(self, position: OpenPosition, bid_price: float) -> OrderPlan:
        """Orden de venta con ajuste de latencia."""
        limit_price = round(bid_price - config.LATENCY_OFFSET_CENTS, 4)
        return OrderPlan(
            symbol=position.symbol, side="SELL", limit_price=limit_price,
            qty=position.qty, stop_loss=0, take_profit=0,
            min_profit=config.TARGET_MIN_NET_PROFIT_USD,
            is_viable=True,
        )

    # ── EVALUACIÓN DE SALIDA ───────────────────────────────────────────────
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
        position = self._update_trailing_stop(position, current_price)

        if current_price <= position.stop_loss:
            pnl = position.pnl(current_price)
            is_trailing = position.stop_loss > position.initial_stop
            label = "🔒 TRAILING_STOP" if is_trailing else "🛑 STOP_LOSS"
            return True, (
                f"{label} alcanzado "
                f"(${current_price:.2f} ≤ ${position.stop_loss:.2f}) "
                f"PnL={pnl:+.2f}"
            )

        if current_price >= position.take_profit:
            pnl = position.pnl(current_price)
            return True, (
                f"🎯 TAKE_PROFIT alcanzado "
                f"(${current_price:.2f} ≥ ${position.take_profit:.2f}) "
                f"PnL={pnl:+.2f}"
            )

        minutes = position.minutes_open
        if minutes >= self.MAX_MINUTES_IN_POSITION:
            pnl = position.pnl(current_price)
            return True, (
                f"⏰ CIERRE_POR_TIEMPO "
                f"({minutes:.0f} min ≥ {self.MAX_MINUTES_IN_POSITION} min) "
                f"PnL={pnl:+.2f} — capital liberado para otra oportunidad"
            )

        return False, ""

    # ── TRAILING STOP INTERNO ──────────────────────────────────────────────
    def _update_trailing_stop(
        self, position: OpenPosition, current_price: float
    ) -> OpenPosition:
        """
        Mueve el Stop Loss hacia arriba cuando el precio sube.
        Se activa cuando la posición está en ganancia de al menos 1%.
        """
        if current_price < position.entry_price * 1.01:
            return position

        if current_price > position.highest_price:
            position.highest_price = current_price
            original_sl_distance = position.entry_price - position.initial_stop
            new_sl = round(position.highest_price - original_sl_distance, 4)
            if new_sl > position.stop_loss:
                old_sl = position.stop_loss
                position.stop_loss = new_sl
                log.debug(
                    f"🔒 TRAILING_STOP | {position.symbol} | "
                    f"Nuevo máximo=${current_price:.2f} → "
                    f"SL: ${old_sl:.2f} → ${new_sl:.2f}"
                )

        return position
