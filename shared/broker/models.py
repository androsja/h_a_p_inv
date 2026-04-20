"""
broker/models.py ─ Modelos de datos para el Broker Simulado.

Contiene estructuras como PendingOrder y SessionStats para desacoplar
la lógica de ejecución del simulador de su estado y métricas.
"""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
import numpy as np

# ─── Orden pendiente en el simulador ────────────────────────────────────────
@dataclass
class PendingOrder:
    order_id:    str
    symbol:      str
    side:        str
    limit_price: float
    qty:         float
    order_type:  str = "LIMIT" # "LIMIT" or "STOP"
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
    gross_profit:   float = 0.0        # Suma de trades con Gross PnL > 0
    gross_loss:     float = 0.0        # Suma de trades con Gross PnL <= 0
    total_slippage: float = 0.0        # Costo real acumulado por deslizamiento (slippage)
    net_profit:     float = 0.0        # Suma de trades con Net PnL > 0
    net_loss:       float = 0.0        # Suma de trades con Net PnL <= 0
    peak_balance:   float = 10_000.0   # Para calcular Max Drawdown
    max_drawdown:   float = 0.0        # Peor caída desde el máximo (%)
    _pnl_history:   list  = field(default_factory=list)  # PnL por trade
    trade_history:  list  = field(default_factory=list)  # Lista de dicts con detalle de cada trade

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round((self.winning_trades / self.total_trades) * 100, 2)

    @property
    def profit_factor(self) -> float:
        """Ratio ganancia bruta / pérdida bruta. >1.5 = bueno, >2.0 = excelente."""
        if self.gross_loss == 0:
            return 999.0 if self.gross_profit > 0 else 0.0
        return abs(self.gross_profit / self.gross_loss)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe Ratio anualizado (simplificado para intraday)."""
        if len(self._pnl_history) < 2:
            return 0.0
        returns = np.array(self._pnl_history)
        mean_r  = returns.mean()
        std_r   = returns.std()
        if std_r == 0:
            return 0.0
        # Anualizado: ~1260 trades/año
        return float((mean_r / std_r) * (1260 ** 0.5))

    def kelly_fraction(self, rr_ratio: float = 2.0) -> float:
        """Criterio de Kelly Fraccionado."""
        if self.total_trades < 10:
            return 0.01
        p = self.win_rate / 100
        q = 1 - p
        full_kelly = (rr_ratio * p - q) / rr_ratio
        return max(0.005, min(0.02, full_kelly * 0.25))

    def record_trade(self, gross_pnl: float, net_pnl: float, current_balance: float, slippage: float = 0.0, side: str = "SELL", qty: float = 0.0, price: float = 0.0, timestamp: str = "", reason: str = "", metadata: dict = None) -> None:
        self.total_trades    += 1
        self.total_pnl       += net_pnl
        self.total_slippage  += slippage
        self._pnl_history.append(net_pnl)
        
        # Append to detailed history
        self.trade_history.append({
            "side": side,
            "qty": qty,
            "price": price,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "pnl": net_pnl, # por compatibilidad
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat()
        })
        
        # Clasificar como Win/Loss basado en PROFIT BRUTO (antes de comisiones)
        # Esto asegura que el Win Rate refleje la efectividad de la estrategia
        if gross_pnl > 0:
            self.winning_trades += 1
            self.gross_profit   += gross_pnl
        else:
            self.gross_loss += gross_pnl
            
        # También llevar cuenta del Neto para el balance final
        if net_pnl > 0:
            self.net_profit += net_pnl
        else:
            self.net_loss += net_pnl

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
