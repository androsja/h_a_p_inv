"""
broker/hapi_mock.py ─ Implementación de simulación usando MarketReplay.

Simula un bróker real reproduciendo datos históricos de yfinance barra a barra.
Permite probar la estrategia completa sin arriesgar capital real.

Comportamiento:
  • Las quotes se obtienen de la vela histórica actual del replay.
  • Las órdenes límite se "ejecutan" si el precio del replay cruza el límite.
  • El saldo de la cuenta es virtual y se actualiza con cada operación.
  • El settlement T+1 se simula correctamente (bloqueando el cash vendido).
  • Las estadísticas finales muestran PnL total, win rate y número de trades.
"""

import uuid
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared import config
from shared.broker.interface import BrokerInterface, Quote, OrderResponse, AccountInfo
from shared.data.market_data import MarketReplay, LivePaperReplay
from shared.utils.logger import log, log_order_filled


# ─── Orden pendiente en el simulador ────────────────────────────────────────
@dataclass
class PendingOrder:
    order_id:    str
    symbol:      str
    side:        str
    limit_price: float
    qty:         float
    created_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason:      str = ""
    metadata:    dict = field(default_factory=dict)


# ─── Estadísticas de la sesión simulada ─────────────────────────────────────
@dataclass
class SessionStats:
    total_trades:   int   = 0
    winning_trades: int   = 0
    total_pnl:      float = 0.0
    total_fees:     float = 0.0
    gross_profit:   float = 0.0
    gross_loss:     float = 0.0
    peak_balance:   float = 10_000.0   # Para calcular Max Drawdown
    max_drawdown:   float = 0.0        # Peor caída desde el máximo (%)
    _pnl_history:   list  = field(default_factory=list)  # PnL por trade

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        """Ratio ganancia bruta / pérdida bruta. >1.5 = bueno, >2.0 = excelente."""
        if self.gross_loss == 0:
            return 999.0 if self.gross_profit > 0 else 0.0
        return abs(self.gross_profit / self.gross_loss)

    @property
    def sharpe_ratio(self) -> float:
        """
        Sharpe Ratio anualizado (simplificado para intraday).
        Ratio retorno/riesgo. >1.0 = aceptable, >2.0 = bueno.
        """
        if len(self._pnl_history) < 2:
            return 0.0
        returns = np.array(self._pnl_history)
        mean_r  = returns.mean()
        std_r   = returns.std()
        if std_r == 0:
            return 0.0
        # Anualizado: ~252 días × ~5 trades/día = ~1260 trades/año
        return float((mean_r / std_r) * (1260 ** 0.5))

    def kelly_fraction(self, rr_ratio: float = 2.0) -> float:
        """
        Criterio de Kelly Fraccionado (25% del Kelly completo).
        Determina el % óptimo del capital a arriesgar por trade.

        f* = (b×p - q) / b
        donde b=rr_ratio, p=win_rate, q=1-p

        Args:
            rr_ratio: relación beneficio/riesgo (TP/SL, default 2.0)
        """
        if self.total_trades < 10:
            return 0.01   # Antes de 10 trades, arriesgar 1% fijo (conservador)
        p = self.win_rate / 100
        q = 1 - p
        full_kelly = (rr_ratio * p - q) / rr_ratio
        # Fractional Kelly: 25% del kelly completo (más seguro)
        frac_kelly = max(0.005, min(0.02, full_kelly * 0.25))
        return frac_kelly

    def record_trade(self, pnl: float, current_balance: float) -> None:
        self.total_trades += 1
        self.total_pnl    += pnl
        self._pnl_history.append(pnl)
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit   += pnl
        else:
            self.gross_loss += pnl
        # Actualizar Max Drawdown
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  📊 RESUMEN DE SESIÓN SIMULADA (HAPI SHADOW)\n"
            f"{'='*55}\n"
            f"  Trades totales : {self.total_trades}\n"
            f"  Win rate       : {self.win_rate:.1f}%\n"
            f"  PnL NETO       : ${self.total_pnl:+.2f}\n"
            f"  IBKR Fees Paid : ${self.total_fees:.2f}\n"
            f"  Ganancia bruta : ${self.gross_profit:.2f}\n"
            f"  Pérdida bruta  : ${self.gross_loss:.2f}\n"
            f"  Profit Factor  : {self.profit_factor:.2f}\n"
            f"  Sharpe Ratio   : {self.sharpe_ratio:.2f}\n"
            f"  Max Drawdown   : {self.max_drawdown:.1f}%\n"
            f"{'='*55}"
        )


class HapiMock(BrokerInterface):
    """
    Bróker simulado que reproduce datos históricos de yfinance.

    Flujo de simulación:
      1. MarketReplay descarga datos de 1 minuto de un símbolo.
      2. Cada llamada a get_quote() avanza una vela.
      3. Las órdenes límite se comprueban contra el precio de la siguiente vela.
      4. El capital virtual se actualiza con cada fill.
    """

    def __init__(self, symbol: str | None = None, initial_cash: float = 10_000.0, live_paper: bool = False, start_date: str | None = None):
        self._live_paper     = live_paper
        self._replay         = LivePaperReplay(symbol) if live_paper else MarketReplay(symbol, start_date=start_date)
        self._cash           = initial_cash
        self._initial_cash   = initial_cash
        self._pending_orders: list[PendingOrder] = []
        self._current_bar: pd.Series | None       = None
        self._stats          = SessionStats()
        # Settlement: {fecha_str: amount}
        self._pending_settlement: dict[str, float] = {}
        log.info(
            f"HapiMock | Simulando {self._replay.symbol} | "
            f"Saldo inicial: ${initial_cash:,.2f}"
        )

    # ─── Propiedades ─────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "HapiMock"

    @property
    def is_paper_trading(self) -> bool:
        return True

    @property
    def symbol(self) -> str:
        return self._replay.symbol

    @property
    def has_more_data(self) -> bool:
        return self._replay.has_next()

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def available_cash(self) -> float:
        """Capital disponible (sin contar el bloqueado por T+1 simulation)."""
        blocked = sum(self._pending_settlement.values())
        return max(0.0, self._cash - blocked)

    # ─── Implementación de BrokerInterface ───────────────────────────────────
    def get_quote(self, symbol: str) -> Quote:
        """
        Avanza una vela del replay y retorna los precios de esa vela.
        bid = Open de la vela (precio de venta más conservador)
        ask = Close de la vela (precio de compra más actualizado)
        """
        from shared.utils.market_hours import _is_mock_time_active
        
        if not self._replay.has_next():
            raise StopIteration("No quedan más datos históricos para simular.")

        try:
            new_bar = self._replay.next_bar()
            
            # Si en Test Nocturno el reloj acelerado aún no ha revelado una nueva vela
            if new_bar is None:
                if self._current_bar is None:
                    raise ConnectionError("Esperando la primera vela del día...")
                # Retenemos la vela anterior si todavía no hay una nueva
            elif _is_mock_time_active() and self._current_bar is not None:
                if new_bar.name == self._current_bar.name:
                    pass # Mantenemos el _current_bar anterior, no avanzamos "tiempo"
                else:
                    self._current_bar = new_bar
            else:
                self._current_bar = new_bar
                
        except Exception as e:
            if _is_mock_time_active() and isinstance(e, ConnectionError):
                 raise e # Dejar pasar el ConnectionError para que main.py duerma 5s
            elif _is_mock_time_active() and self._current_bar is not None:
                log.debug(f"MockTime: Alpaca rate limit o espera. Usando barra anterior. ({e})")
            else:
                log.error(f"HapiMock StopIteration originado por: {type(e).__name__} - {str(e)}")
                raise StopIteration(f"No quedan más datos históricos para simular. ({e})")
        
        bar = self._current_bar

        # Simular bid/ask razonable: bid ligeramente por debajo del close,
        # ask ligeramente por encima (spread de 1 centavo)
        close = float(bar["Close"])
        bid   = round(close - 0.005, 4)
        ask   = round(close + 0.005, 4)

        # Intentar rellenar órdenes pendientes con el precio de esta vela
        self._try_fill_pending_orders(bar)

        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=close,
            timestamp=str(bar.name) if hasattr(bar, "name") else datetime.now(timezone.utc).isoformat(),
        )

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
        reason: str = "",
        metadata: dict = None,
    ) -> OrderResponse:
        """
        Registra una orden límite pendiente.
        La orden se "ejecuta" cuando el precio del replay es favorable.
        """
        if qty <= 0:
            return OrderResponse(
                order_id="REJECTED",
                symbol=symbol, side=side,
                status="REJECTED",
                limit_price=limit_price, qty=qty,
                fill_price=None,
                message="Cantidad inválida (<= 0)",
            )

        order = PendingOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side,
            limit_price=limit_price, qty=qty,
            reason=reason,
            metadata=metadata or {},
        )
        self._pending_orders.append(order)
        log.debug(
            f"HapiMock | {side} {qty:.4f} {symbol} @ límite ${limit_price:.2f} → PENDIENTE"
        )
        return OrderResponse(
            order_id=order.order_id,
            symbol=symbol, side=side,
            status="PENDING",
            limit_price=limit_price, qty=qty,
            fill_price=None,
        )

    def get_account_info(self) -> AccountInfo:
        last_p = float(self._current_bar['Close']) if self._current_bar is not None else 0.0
        return AccountInfo(
            total_cash=self._cash,
            buying_power=self.available_cash,
            portfolio_value=self._cash,
            day_traded_count=0,
            last_price=last_p
        )

    def cancel_all_orders(self, symbol: str) -> bool:
        before = len(self._pending_orders)
        self._pending_orders = [o for o in self._pending_orders if o.symbol != symbol]
        cancelled = before - len(self._pending_orders)
        if cancelled:
            log.info(f"HapiMock | {cancelled} orden(es) de {symbol} canceladas.")
        return True

    def immediate_market_sell(self, symbol: str, qty: float, bid_price: float, reason: str = "", metadata: dict = None) -> float:
        """
        Venta de mercado INMEDIATA — se ejecuta al bid_price actual.
        SOLO para Stop Loss, Trailing Stop y Time Exit.
        Garantiza salida inmediata sin esperar que el precio regrese.
        Devuelve el precio de fill real.
        """
        slippage   = 0.0005
        fill_price = round(bid_price * (1 - slippage), 2)
        fake_order = PendingOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side="SELL",
            limit_price=fill_price, qty=qty,
            reason=reason,
            metadata=metadata or {},
        )
        self._execute_sell(fake_order, fill_price=fill_price)
        log.info(
            f"HapiMock | ⚡ MARKET SELL ejecutado para {symbol}: "
            f"{qty:.4f} acciones @ ${fill_price:.2f} (bid={bid_price:.2f})"
        )
        return fill_price

    def get_history_slice(self, window: int = 220) -> pd.DataFrame:
        """Retorna las últimas `window` velas para el cálculo de indicadores.
        220 = EMA 200 + 20 de margen para estabilizar los cálculos."""
        return self._replay.current_slice(window)

    # ─── Motor de ejecución del simulador ─────────────────────────────────────
    def _try_fill_pending_orders(self, bar: pd.Series) -> None:
        """
        Intenta ejecutar órdenes pendientes contra los precios de la vela actual.

        Lógica:
          • BUY  LIMIT → se ejecuta si Low de la vela ≤ limit_price (precio fue alcanzado)
          • SELL LIMIT → se ejecuta si High de la vela ≥ limit_price
        """
        filled: list[PendingOrder] = []
        bar_low  = float(bar["Low"])
        bar_high = float(bar["High"])
        bar_vol  = float(bar.get("Volume", 0))

        # Requisito de simulación realista: la orden no puede tomar más del 10% del volumen del minuto.
        # En caso de no haber datos de volumen (0), permitimos que pase para evitar bloquear ETFs raros o testing.
        max_fillable_qty = bar_vol * 0.10 if bar_vol > 0 else float('inf')

        for order in self._pending_orders:
            # Control de liquidez:
            if order.qty > max_fillable_qty:
                log.debug(f"HapiMock | Orden por {order.qty} ignorada en vela por falta de liquidez (max {max_fillable_qty})")
                continue
            if order.side == "BUY" and bar_low <= order.limit_price:
                self._execute_buy(order, fill_price=order.limit_price, timestamp=str(bar.name))
                filled.append(order)

            elif order.side == "SELL" and bar_high >= order.limit_price:
                self._execute_sell(order, fill_price=order.limit_price, timestamp=str(bar.name))
                filled.append(order)

        for o in filled:
            self._pending_orders.remove(o)

    def _execute_buy(self, order: PendingOrder, fill_price: float, timestamp: str | None = None) -> None:
        # Slippage simulado de compra (0.05% de penalización por latencia)
        slippage = 0.0005
        actual_fill_price = round(fill_price * (1 + slippage), 2)
        
        # --- IBKR FEES ON BUY (Tiered) ---
        gross_cost = actual_fill_price * order.qty
        comm = max(0.35, min(0.01 * gross_cost, 0.0035 * order.qty))
        cost = gross_cost + comm
        
        if cost > self._cash:
            log.warning(
                f"HapiMock | Orden BUY rechazada: costo ${cost:.2f} > "
                f"cash ${self._cash:.2f}"
            )
            return
        self._cash -= cost
        self._last_buy_price = actual_fill_price
        self._last_buy_qty   = order.qty
        self._stats.total_fees += comm
        
        try:
            from shared.utils.state_writer import _state
            _state.total_fees_paid += comm
        except: pass
        log_order_filled(
            order.symbol, "BUY", actual_fill_price, order.qty, 
            timestamp=timestamp, reason=order.reason, metadata=order.metadata
        )
        # Update global state for UI tracking
        try:
            from shared.utils.state_writer import _state
            _state.available_cash = self._cash
        except: pass

    def _execute_sell(self, order: PendingOrder, fill_price: float, timestamp: str | None = None) -> None:
        # Slippage simulado de venta (0.05% de penalización por latencia)
        slippage = 0.0005
        actual_fill_price = round(fill_price * (1 - slippage), 2)
        
        gross = actual_fill_price * order.qty
        
        # --- IBKR FEES ON SELL (Tiered) ---
        # 1. Comisión accionaria
        comm = max(0.35, min(0.01 * gross, 0.0035 * order.qty))
        
        # 2. SEC Fee ($0.0000278 per $1, min $0.01)
        sec_fee = max(0.01, round(gross * 0.0000278, 2))
        
        # 3. FINRA TAF ($0.000166 per share, max $5.95)
        taf_fee = min(5.95, round(order.qty * 0.000166, 2))
        if taf_fee < 0.01: 
            taf_fee = 0.01 # minimal
            
        total_fees = comm + sec_fee + taf_fee
        
        net = gross - total_fees
        self._cash += net
        self._stats.total_fees += total_fees
        
        raw_pnl = (actual_fill_price - getattr(self, '_last_buy_price', actual_fill_price)) * order.qty
        net_pnl = raw_pnl - total_fees
        
        log_order_filled(
            order.symbol, "SELL", actual_fill_price, order.qty, net_pnl, 
            timestamp=timestamp, reason=order.reason, metadata=order.metadata
        )
        log.info(f"🏦 IBKR FEES COBRADOS: Com=${comm:.2f} | SEC=${sec_fee:.4f} | TAF=${taf_fee:.4f} | Total=${total_fees:.2f}")
        
        self._stats.record_trade(net_pnl, current_balance=self._cash)
        
        # Update global state
        try:
            from shared.utils.state_writer import _state
            _state.available_cash = self._cash
            _state.net_profit += net_pnl
            _state.total_fees_paid += total_fees
        except: pass

    def print_stats(self) -> None:
        """Imprime el resumen final de la sesión simulada."""
        final_pnl = self._cash - self._initial_cash
        log.info(
            f"HapiMock | Saldo final: ${self._cash:.2f} | "
            f"PnL total: ${final_pnl:+.2f} vs inicio ${self._initial_cash:.2f}"
        )
        print(self._stats.summary())
