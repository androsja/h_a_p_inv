"""
broker/interface.py ─ Clase abstracta BrokerInterface.

Define el contrato que TODAS las implementaciones de bróker deben cumplir.
Esto garantiza que el motor principal del bot puede funcionar con cualquier
bróker (HapiLive, HapiMock, o futuras implementaciones) sin cambiar la
lógica de negocio.

Métodos obligatorios:
  • get_quote(symbol) → Quote
  • place_limit_order(plan) → OrderResponse
  • get_account_info()  → dict con saldo disponible
  • cancel_all_orders(symbol) → bool
  • get_open_positions() → list[PositionInfo]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ─── Tipos de datos de respuesta ────────────────────────────────────────────
@dataclass
class Quote:
    """Precio actual de un instrumento."""
    symbol: str
    bid: float          # Precio de VENTA del mercado (lo que recibirías)
    ask: float          # Precio de COMPRA del mercado (lo que pagarías)
    last: float         # Último precio transado
    timestamp: str      # ISO 8601

    @property
    def mid(self) -> float:
        """Precio medio (referencia)."""
        return round((self.bid + self.ask) / 2, 4)

    @property
    def spread(self) -> float:
        """Spread bid-ask en dólares."""
        return round(self.ask - self.bid, 4)


@dataclass
class OrderResponse:
    """Respuesta del bróker al colocar una orden."""
    order_id:    str
    symbol:      str
    side:        str           # "BUY" | "SELL"
    status:      str           # "PENDING" | "FILLED" | "CANCELLED" | "REJECTED"
    limit_price: float
    qty:         float
    fill_price:  float | None  # Precio real de ejecución (None si aún no se ejecuta)
    message:     str = ""      # Mensaje del bróker (útil para errores)


@dataclass
class AccountInfo:
    """Estado de la cuenta del bróker."""
    total_cash:          float
    buying_power:        float     # Capital disponible para nuevas órdenes
    portfolio_value:     float
    day_traded_count:    int = 0   # Número de operaciones en el día


# ─── Interfaz abstracta ──────────────────────────────────────────────────────
class BrokerInterface(ABC):
    """
    Contrato abstracto para cualquier implementación de bróker.

    Toda clase que extienda BrokerInterface DEBE implementar todos los
    métodos marcados con @abstractmethod. El motor del bot solo interactúa
    con esta interfaz, nunca directamente con HapiLive o HapiMock.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del bróker (ej. 'HapiLive', 'HapiMock')."""
        ...

    @property
    @abstractmethod
    def is_paper_trading(self) -> bool:
        """True si es un entorno de prueba (no usa dinero real)."""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Obtiene la cotización actual de un símbolo.

        Returns:
            Quote con bid, ask, last y timestamp.
        Raises:
            ConnectionError: Si no puede contactar al bróker.
            ValueError: Si el símbolo no es válido.
        """
        ...

    @abstractmethod
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        limit_price: float,
        qty: float,
    ) -> OrderResponse:
        """
        Coloca una orden límite en el bróker.

        Args:
            symbol:      Símbolo del instrumento (ej. 'AAPL').
            side:        'BUY' o 'SELL'.
            limit_price: Precio máximo (compra) o mínimo (venta) aceptado.
            qty:         Cantidad de acciones (puede ser fraccionario).

        Returns:
            OrderResponse con el estado de la orden.
        """
        ...

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Retorna el estado actual de la cuenta (saldos, buying power)."""
        ...

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancela todas las órdenes abiertas para un símbolo.

        Returns:
            True si todas se cancelaron exitosamente.
        """
        ...

    # ─── Método opcional con implementación base ─────────────────────────────
    def validate_symbol(self, symbol: str) -> bool:
        """
        Valida que el símbolo esté en el listado de activos permitidos.
        Las subclases pueden sobreescribir esto para validar contra la API.
        """
        import json
        from shared import config
        with open(config.ASSETS_FILE) as f:
            assets = json.load(f)["assets"]
        allowed = {a["symbol"] for a in assets}
        return symbol in allowed
