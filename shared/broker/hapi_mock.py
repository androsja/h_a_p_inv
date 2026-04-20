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
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
import requests
import os

from shared import config
from shared.broker.interface import BrokerInterface, Quote, OrderResponse, AccountInfo
from shared.broker.models    import PendingOrder, SessionStats
from shared.data.replay      import MarketReplay, LivePaperReplay
from shared.utils.logger     import log, log_order_filled



# ─── HAPI MOCK BROKER ───────────────────────────────────────────────────────


class HapiMock(BrokerInterface):
    """
    Bróker simulado que reproduce datos históricos de yfinance.

    Flujo de simulación:
      1. MarketReplay descarga datos de 1 minuto de un símbolo.
      2. Cada llamada a get_quote() avanza una vela.
      3. Las órdenes límite se comprueban contra el precio de la siguiente vela.
      4. El capital virtual se actualiza con cada fill.
    """

    def __init__(self, symbol: Optional[str] = None, initial_cash: float = 10_000.0, live_paper: bool = False, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self._live_paper     = live_paper
        self._replay         = LivePaperReplay(symbol) if live_paper else MarketReplay(symbol, start_date=start_date, end_date=end_date)
        self._cash           = initial_cash
        self._initial_cash   = initial_cash
        self._pending_orders: List[PendingOrder] = []
        self._current_bar: Optional[pd.Series]       = None
        self._stats          = SessionStats()
        # Mantenimiento de precios de entrada nominales para contabilidad transparente (PnL Bruto)
        self._nominal_entry_prices: Dict[str, float] = {}
        
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
            is_mock_active = _is_mock_time_active()
            
            # --- LÓGICA DE AVANCE DIFERENCIADA ---
            if is_mock_active:
                from shared.utils.market_hours import now_nyc
                import pytz
                current_mock_time = now_nyc().astimezone(pytz.UTC)
                
                # REGLA 1: Si ya llegamos a las 4:00 PM NYC, terminamos la sesión
                from datetime import time as dt_time
                if now_nyc().time() >= dt_time(16, 0):
                    log.info("🏁 Mock Time alcanzó las 4:00 PM (Cierre). Finalizando sesión.")
                    raise StopIteration("Cierre de mercado mock alcanzado.")

                # REGLA 2: Solo avanzar si la siguiente vela ya aconteció en el reloj mock
                if hasattr(self._replay, "peek_next_timestamp"):
                    next_ts = self._replay.peek_next_timestamp()
                    if next_ts and next_ts > current_mock_time:
                        # Todavía no es hora de la siguiente vela. Retenemos barra actual.
                        if self._current_bar is None:
                            # Si es el arranque, forzamos la primera para no colgar el motor
                            self._current_bar = self._replay.next_bar()
                        return self._generate_quote(self._current_bar, symbol)
                
            # Avance normal (o forzado por falta de Mock)
            new_bar = self._replay.next_bar()
            
            if new_bar is None:
                if self._current_bar is None:
                    raise ConnectionError("Esperando la primera vela del día...")
            elif is_mock_active and self._current_bar is not None:
                if new_bar.name == self._current_bar.name:
                    pass # Mantenemos el _current_bar anterior, no avanzamos "tiempo"
                else:
                    self._current_bar = new_bar
            else:
                self._current_bar = new_bar
                
        except StopIteration:
            raise StopIteration("No quedan más datos históricos para simular.")
        except Exception as e:
            if _is_mock_time_active() and isinstance(e, ConnectionError):
                 raise e # Dejar pasar el ConnectionError para que main.py duerma 5s
            elif _is_mock_time_active() and self._current_bar is not None:
                log.debug(f"MockTime: Alpaca rate limit o espera. Usando barra anterior. ({e})")
            else:
                log.error(f"HapiMock StopIteration originado por: {type(e).__name__} - {str(e)}")
                raise StopIteration(f"No quedan más datos históricos para simular. ({e})")
        
        bar = self._current_bar
        return self._generate_quote(bar, symbol)

    def _generate_quote(self, bar: pd.Series, symbol: str) -> Quote:
        """Centraliza la generación de Quote con lógica de spread/slippage."""
        close_p = float(bar["Close"])
        high_p  = float(bar["High"])
        low_p   = float(bar["Low"])
        
        # 10% de la volatilidad intraburbuja, con piso de 0.02c (bluechips tranquilas)
        volatility_spread = (high_p - low_p) * 0.10
        spread_val = round(max(0.02, volatility_spread), 3)

        bid    = round(close_p - (spread_val / 2), 4)
        ask    = round(close_p + (spread_val / 2), 4)

        # Intentar rellenar órdenes pendientes con el precio de esta vela
        self._try_fill_pending_orders(bar)

        from datetime import datetime, timezone
        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=close_p,
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
        Registra una orden límite o stop pendiente.
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

        order_type = "LIMIT"
        if reason and ("STOP_LOSS" in reason or "TRAILING_STOP" in reason):
            order_type = "STOP"

        order = PendingOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side,
            limit_price=limit_price, qty=qty,
            order_type=order_type,
            reason=reason,
            metadata=metadata or {},
        )
        self._pending_orders.append(order)
        log.debug(
            f"HapiMock | {side} {qty:.4f} {symbol} [{order_type}] @ ${limit_price:.2f} → PENDIENTE ({reason})"
        )
        return OrderResponse(
            order_id=order.order_id,
            symbol=symbol, side=side,
            status="PENDING",
            limit_price=limit_price, qty=qty,
            fill_price=None,
        )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reason: str = "",
        metadata: dict = None,
    ) -> OrderResponse:
        """
        Ejecución inmediata a precio de mercado (usando la vela actual).
        """
        import uuid
        from shared.broker.models import PendingOrder

        if self._current_bar is None:
            # Si se llama antes del primer get_quote, intentamos obtener uno
            self.get_quote(symbol)

        bar = self._current_bar
        # Simular ejecución al precio close de la vela actual
        fill_price = float(bar["Close"])
        
        order = PendingOrder(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol, side=side.upper(),
            limit_price=fill_price, qty=qty,
            reason=reason,
            metadata=metadata or {},
        )

        if side.upper() == "BUY":
            self._execute_buy(order, fill_price=fill_price)
        else:
            self._execute_sell(order, fill_price=fill_price)

        return OrderResponse(
            order_id=order.order_id,
            symbol=symbol, side=side.upper(),
            status="FILLED",
            limit_price=0, qty=qty,
            fill_price=fill_price,
            message=reason
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
        # No aplicamos slippage aquí, se aplicará una única vez en _execute_sell
        fill_price = bid_price
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

    def get_bars_df(self, symbol: str, limit: int = 220) -> pd.DataFrame:
        """
        Retorna las últimas `limit` velas desde el replay.
        Mismo comportamiento que AlpacaPaperBroker pero usando datos históricos.
        """
        return self._replay.current_slice(limit)

    def get_bars_df(self, symbol: str, limit: int = 220) -> pd.DataFrame:
        """
        Retorna las últimas `limit` velas desde el replay.
        Mismo comportamiento que AlpacaPaperBroker pero usando datos históricos.
        """
        return self._replay.current_slice(limit)

    def get_history_slice(self, window: int = 220) -> pd.DataFrame:
        """Retorna las últimas `window` velas para el cálculo de indicadores.
        220 = EMA 200 + 20 de margen para estabilizar los cálculos."""
        return self._replay.current_slice(window)

    # ─── Motor de ejecución del simulador ─────────────────────────────────────
    def _try_fill_pending_orders(self, bar: pd.Series) -> None:
        """
        Intenta ejecutar órdenes pendientes contra los precios de la vela actual.
        Diferencia entre órdenes LIMIT (mejorar precio) y STOP (protección).
        """
        filled: list[PendingOrder] = []
        bar_low   = float(bar["Low"])
        bar_high  = float(bar["High"])
        bar_open  = float(bar["Open"])
        bar_vol   = float(bar.get("Volume", 0))

        # Requisito de simulación realista: la orden no puede tomar más del 10% del volumen del minuto.
        max_fillable_qty = bar_vol * 0.10 if bar_vol > 0 else float('inf')

        for order in self._pending_orders:
            if order.qty > max_fillable_qty:
                log.debug(f"HapiMock | Orden {order.symbol} ignorada por liquidez (max {max_fillable_qty})")
                continue

            if order.side == "BUY":
                if order.order_type == "LIMIT":
                    if bar_low <= order.limit_price:
                        # Comprar si toca el límite por debajo (Límite normal)
                        self._execute_buy(order, fill_price=order.limit_price, timestamp=str(bar.name))
                        filled.append(order)
                else: # STOP BUY
                    if bar_high >= order.limit_price:
                        # Comprar si rompe al alza (Stop de compra)
                        fill_p = max(order.limit_price, bar_open)
                        self._execute_buy(order, fill_price=fill_p, timestamp=str(bar.name))
                        filled.append(order)

            elif order.side == "SELL":
                if order.order_type == "LIMIT":
                    if bar_high >= order.limit_price:
                        # Venta favorable
                        self._execute_sell(order, fill_price=order.limit_price, timestamp=str(bar.name))
                        filled.append(order)
                else: # STOP SELL (Stop Loss / Trailing Stop)
                    if bar_low <= order.limit_price:
                        # Venta de protección: se activa cuando el precio CAE al límite
                        fill_p = min(order.limit_price, bar_open)
                        self._execute_sell(order, fill_price=fill_p, timestamp=str(bar.name))
                        filled.append(order)

        for o in filled:
            self._pending_orders.remove(o)

    def _execute_buy(self, order: PendingOrder, fill_price: float, timestamp: Optional[str] = None) -> None:
        # Slippage simulado de compra (0.05% de penalización por latencia)
        slippage = 0.0005
        actual_fill_price = round(fill_price * (1 + slippage), 2)
        
        # --- IBKR FEES ON BUY (Tiered) ---
        gross_cost = actual_fill_price * order.qty
        comm = max(0.35, min(0.01 * gross_cost, 0.0035 * order.qty))
        cost = gross_cost + comm
        
        # --- Orchestrator Bank Integration ---
        orchestrator_url = os.getenv("ORCHESTRATOR_URL")
        if orchestrator_url:
            try:
                resp = requests.post(f"{orchestrator_url}/api/bank/transaction", 
                                     json={"symbol": order.symbol, "amount": float(cost), "type": "buy"}, 
                                     timeout=3)
                if resp.status_code != 200:
                    log.warning(f"HapiMock | Orden BUY rechazada por Banco Central: {resp.text}")
                    return
                # Sincronizar cash local con el devuelto por el banco por si acaso
                self._cash = resp.json().get("available_cash", self._cash)
            except Exception as e:
                log.warning(f"Error contactando al Banco Central: {e}")
                return
        else:
            if cost > self._cash:
                log.warning(
                    f"HapiMock | Orden BUY rechazada: costo ${cost:.2f} > "
                    f"cash ${self._cash:.2f}"
                )
                return
            self._cash -= cost
            
        self._last_buy_price     = actual_fill_price
        self._nominal_entry_prices[order.symbol] = fill_price
        self._last_buy_slippage  = round(fill_price * slippage * order.qty, 2)
        self._last_buy_qty       = order.qty
        self._stats.total_fees  += comm
        
        # Record Buy in history for "Last Order" tracking
        self._stats.trade_history.append({
            "side": "BUY",
            "price": actual_fill_price,
            "qty": order.qty,
            "reason": order.reason,
            "metadata": order.metadata,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat()
        })
        
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

    def _execute_sell(self, order: PendingOrder, fill_price: float, timestamp: Optional[str] = None) -> None:
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
        
        # --- Orchestrator Bank Integration ---
        orchestrator_url = os.getenv("ORCHESTRATOR_URL")
        if orchestrator_url:
            try:
                resp = requests.post(f"{orchestrator_url}/api/bank/transaction", 
                                     json={"symbol": order.symbol, "amount": float(net), "type": "sell"}, 
                                     timeout=3)
                if resp.status_code == 200:
                    self._cash = resp.json().get("available_cash", self._cash)
            except Exception as e:
                log.warning(f"Error contactando Banco Central en SELL: {e}")
        else:
            self._cash += net
            
        self._stats.total_fees += total_fees
        
        # Cálculos de contabilidad transparente
        # 1. PnL Nominal (Puro mercado - lo que el usuario ve en el calendario)
        nominal_entry_price = self._nominal_entry_prices.get(order.symbol, fill_price)
        nominal_exit_price  = fill_price # El precio de la vela sin slippage
        nominal_pnl         = round((nominal_exit_price - nominal_entry_price) * order.qty, 2)

        # 2. Slippage real (La penalización por ejecución)
        slippage_pct = 0.0005
        buy_slip = getattr(self, '_last_buy_slippage', 0.0)
        sell_slip = round(fill_price * slippage_pct * order.qty, 2)
        total_trade_slippage = round(buy_slip + sell_slip, 2)

        # 3. PnL Neto (Lo que realmente queda en el banco)
        net_pnl = round(nominal_pnl - total_trade_slippage - total_fees, 2)
        
        # USAR PNL NOMINAL (BRUTO) EN EL DIARIO PARA TRANSPARENCIA TOTAL
        # El usuario sumará $15 + $16 y le dará los +$31 de la tarjeta superior.
        log_order_filled(
            order.symbol, "SELL", actual_fill_price, order.qty, nominal_pnl, 
            timestamp=timestamp, reason=order.reason, metadata=order.metadata
        )
        log.info(f"💰 CONTABILIDAD TRANSPARENTE | {order.symbol}: Gross=${nominal_pnl:+.2f} | Slip=${total_trade_slippage:.2f} | Fees=${total_fees:.2f} | Net=${net_pnl:+.2f}")
        
        self._stats.record_trade(
            gross_pnl=nominal_pnl, 
            net_pnl=net_pnl, 
            slippage=total_trade_slippage,
            current_balance=self._cash, 
            side="SELL", 
            qty=order.qty, 
            price=actual_fill_price, 
            timestamp=timestamp, 
            reason=order.reason, 
            metadata=order.metadata
        )
        
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
