"""
shared/broker/alpaca_paper.py ─ Broker de Alpaca Paper Trading.

Implementa BrokerInterface usando la API oficial de Alpaca (alpaca-py).
Este archivo es completamente independiente de HapiMock y HapiLive.
NO toca ni modifica nada de la simulación.

Paper Trading URL: https://paper-api.alpaca.markets
"""

import time
from datetime import datetime, timezone
from typing import Optional

from shared.broker.interface import BrokerInterface, Quote, OrderResponse, AccountInfo
from shared.utils.logger import log


class AlpacaPaperBroker(BrokerInterface):
    """
    Broker de Alpaca Paper Trading.
    Compras/ventas reales contra la API de Paper de Alpaca.
    Los datos de mercado vienen del Market Data API de Alpaca.
    No afecta en absoluto la simulación histórica.
    """

    def __init__(self, symbol: str, api_key: str, secret_key: str):
        if not api_key or not secret_key:
            raise ValueError(
                "AlpacaPaperBroker requiere ALPACA_API_KEY y ALPACA_SECRET_KEY en .env"
            )

        self.symbol = symbol
        self._api_key = api_key
        self._secret_key = secret_key

        # ── Clientes de alpaca-py ─────────────────────────────────────────────
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.live import StockDataStream

        self._trading = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,            # ← siempre Paper
        )
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        # Stats simples (compatibles con lo que usa main_live para el dashboard)
        self._total_trades   = 0
        self._winning_trades = 0
        self._gross_profit   = 0.0
        self._gross_loss     = 0.0
        self._total_fees     = 0.0
        self._total_slippage = 0.0

        log.info(f"[{symbol}] 🦙 AlpacaPaper | Conectado en modo PAPER ✅")

    # ── Propiedades requeridas por BrokerInterface ────────────────────────────

    @property
    def name(self) -> str:
        return "AlpacaPaper"

    @property
    def is_paper_trading(self) -> bool:
        return True

    # ── Cotización en tiempo real ─────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        """Obtiene bid/ask en tiempo real desde Alpaca."""
        from alpaca.data.requests import StockLatestQuoteRequest
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            resp = self._data_client.get_stock_latest_quote(req)
            q = resp[symbol]
            bid  = float(q.bid_price) if q.bid_price else float(q.ask_price)
            ask  = float(q.ask_price)
            last = (bid + ask) / 2
            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                timestamp=q.timestamp.isoformat() if q.timestamp else datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            log.warning(f"[{symbol}] AlpacaPaper | get_quote error: {e}. Usando fallback de barras.")
            # Fallback: última barra
            try:
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                req2 = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    limit=1,
                )
                bars = self._data_client.get_stock_bars(req2)
                bar = bars[symbol][-1]
                price = float(bar.close)
                return Quote(symbol=symbol, bid=price * 0.9995, ask=price * 1.0005, last=price,
                             timestamp=datetime.now(timezone.utc).isoformat())
            except Exception as e2:
                raise ConnectionError(f"No se pudo obtener cotización de Alpaca para {symbol}: {e2}")

    # ── Datos históricos para indicadores ───────────────────────────────────── 

    def get_bars_df(self, symbol: str, limit: int = 220):
        """
        Descarga las últimas barras directamente desde Alpaca de forma fresca.
        Esto garantiza que en Live Trading jamás usemos datos cacheados
        y siempre tengamos la información más reciente, incluso tras pausas.
        """
        try:
            import pandas as pd
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from datetime import timedelta
            
            # Pedimos los últimos 5 días para asegurar suficientes barras (max 300)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=5)
            
            req = StockBarsRequest(
                symbol_or_symbols=symbol, 
                timeframe=TimeFrame(5, TimeFrameUnit.Minute), 
                start=start_time,
                end=end_time,
                feed="iex" # Usar tier gratuito para evitar fallos de permisos en paper
            )
            bars = self._data_client.get_stock_bars(req)
            
            if bars.df.empty:
                log.warning(f"[{symbol}] Alpaca no devolvió barras. Intentando fallback con yfinance...")
                try:
                    import yfinance as yf
                    yf_df = yf.download(symbol, period="5d", interval="5m", progress=False)
                    if yf_df.empty:
                        raise ValueError(f"Alpaca y yfinance no devolvieron barras para {symbol}")
                    
                    if isinstance(yf_df.columns, pd.MultiIndex):
                        yf_df.columns = yf_df.columns.droplevel(1)
                        
                    raw_df = yf_df.rename(columns={
                        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
                    })
                    # Add missing capitalizations if needed
                    cols_map = {c: c.capitalize() for c in raw_df.columns if c.islower() and c in ["open", "high", "low", "close", "volume"]}
                    if cols_map:
                        raw_df = raw_df.rename(columns=cols_map)
                except Exception as fb_err:
                    raise ValueError(f"Alpaca no devolvió barras y fallback falló: {fb_err}")
            else:
                raw_df = bars.df.xs(symbol) if symbol in bars.df.index.levels[0] else bars.df
                
            raw_df = raw_df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            raw_df.index = pd.to_datetime(raw_df.index, utc=True)
            df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            
            # Solo retornamos las que pedimos de límite si las filtramos localmente (opcional)
            return df.tail(limit)
            
        except Exception as e:
            log.warning(f"[{symbol}] AlpacaPaper | Error solicitando barras en vivo: {e}")
            raise

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
        reason: str = "",
        metadata: dict = None,
    ) -> OrderResponse:
        """Coloca una orden límite en Alpaca Paper."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            req = LimitOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
            resp = self._trading.submit_order(req)
            status = str(resp.status).upper()
            fill_price = float(resp.filled_avg_price) if resp.filled_avg_price else None

            log.info(
                f"[{symbol}] 🦙 AlpacaPaper ORDER | {side} {qty:.4f} @ ${limit_price:.2f} "
                f"→ status={status} | {reason}"
            )

            # Actualizar stats locales
            if "fill" in status.lower() and fill_price:
                self._total_trades += 1

            return OrderResponse(
                order_id=str(resp.id),
                symbol=symbol,
                side=side,
                status=status,
                limit_price=limit_price,
                qty=qty,
                fill_price=fill_price,
                message=reason,
            )
        except Exception as e:
            log.error(f"[{symbol}] AlpacaPaper | Error colocando orden {side}: {e}")
            return OrderResponse(
                order_id="ERROR",
                symbol=symbol, side=side,
                status="REJECTED",
                limit_price=limit_price, qty=qty,
                fill_price=None, message=str(e),
            )

    def place_market_order(self, symbol: str, side: str, qty: float, reason: str = "") -> OrderResponse:
        """Coloca una orden de mercado en Alpaca Paper (para exits de SL/TP)."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=symbol,
                qty=round(qty, 4),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            resp = self._trading.submit_order(req)
            status = str(resp.status).upper()
            fill_price = float(resp.filled_avg_price) if resp.filled_avg_price else None
            log.info(f"[{symbol}] 🦙 MARKET {side} {qty:.4f} → {status}")

            return OrderResponse(
                order_id=str(resp.id), symbol=symbol, side=side,
                status=status, limit_price=fill_price or 0,
                qty=qty, fill_price=fill_price, message=reason,
            )
        except Exception as e:
            log.error(f"[{symbol}] AlpacaPaper market order error: {e}")
            return OrderResponse(
                order_id="ERROR", symbol=symbol, side=side,
                status="REJECTED", limit_price=0, qty=qty,
                fill_price=None, message=str(e),
            )

    def get_account_info(self) -> AccountInfo:
        """Obtiene el balance real de la cuenta Paper de Alpaca."""
        try:
            acct = self._trading.get_account()
            return AccountInfo(
                total_cash=float(acct.cash),
                buying_power=float(acct.buying_power),
                portfolio_value=float(acct.portfolio_value),
                day_traded_count=int(acct.daytrade_count),
                last_price=0.0,
            )
        except Exception as e:
            log.warning(f"AlpacaPaper | get_account_info error: {e}")
            return AccountInfo(total_cash=10000.0, buying_power=10000.0,
                               portfolio_value=10000.0, day_traded_count=0, last_price=0.0)

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancela todas las órdenes abiertas de un símbolo en Alpaca."""
        try:
            all_orders = self._trading.get_orders()
            for order in all_orders:
                if order.symbol == symbol:
                    self._trading.cancel_order_by_id(str(order.id))
            return True
        except Exception as e:
            log.warning(f"[{symbol}] AlpacaPaper | cancel_all_orders: {e}")
            return False

    def get_open_position(self, symbol: str) -> Optional[dict]:
        """Retorna la posición abierta en Alpaca Paper para el símbolo, o None."""
        try:
            pos = self._trading.get_open_position(symbol)
            return {
                "symbol":      pos.symbol,
                "qty":         float(pos.qty),
                "avg_price":   float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
            }
        except Exception:
            return None

    # ── Stats (para compatibilidad con el dashboard) ──────────────────────────

    def update_stats(self, pnl: float, fees: float = 0.0, slippage: float = 0.0):
        """Actualiza stats locales tras cerrar un trade. pnl es el Neto."""
        self._total_trades += 1
        self._total_fees += fees
        self._total_slippage += slippage
        # Reconstruir el Gross PnL
        gross_pnl = pnl + fees + slippage
        if gross_pnl > 0:
            self._winning_trades += 1
            self._gross_profit += gross_pnl
        else:
            self._gross_loss += abs(gross_pnl)

    @property
    def stats(self):
        """Retorna un objeto stats compatible con el dashboard."""
        wins = self._winning_trades
        total = self._total_trades
        return type("Stats", (), {
            "total_trades":   total,
            "winning_trades": wins,
            "win_rate":       round(wins / total * 100, 1) if total else 0.0,
            "gross_profit":   round(self._gross_profit, 2),
            "gross_loss":     round(self._gross_loss, 2),
            "total_pnl":      round(self._gross_profit - self._gross_loss - self._total_fees, 2),
            "profit_factor":  round(self._gross_profit / max(self._gross_loss, 0.01), 2),
            "max_drawdown":   0.0,
            "total_fees":     round(self._total_fees, 2),
            "total_slippage": round(self._total_slippage, 2),
        })()
