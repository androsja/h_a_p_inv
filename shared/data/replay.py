"""
data/replay.py ─ Reproductores de datos históricos para modos de simulación.

Responsabilidad única: reproducir velas históricas vela a vela, desacoplado
de la lógica de descarga que vive en market_data.py.

Clases:
  - MarketReplay     : Replay completo de un período histórico (modo SIM).
  - LivePaperReplay  : Obtiene la vela más reciente desde Alpaca (modo LIVE_PAPER).
"""

import time
import random

import pandas as pd
import pytz

from shared import config
from shared.utils.logger import log
from shared.data.market_data import download_bars, get_symbols

ET = pytz.timezone("America/New_York")


# ════════════════════════════════════════════════════════════════════════════
#  MARKET REPLAY (Modo Simulación)
# ════════════════════════════════════════════════════════════════════════════

class MarketReplay:
    """
    Reproduce los datos históricos de 5 minutos vela a vela.
    El MockBroker usa esta clase para simular el mercado intraday.
    """

    def __init__(self, symbol: str | None = None, start_date: str | None = None):
        """
        Args:
            symbol:     Ticker a simular. Si es None, elige uno al azar de assets.json.
            start_date: Formato "YYYY-MM-DD". Filtra el inicio de la simulación.
        """
        if symbol is None:
            symbols = get_symbols()
            symbol = random.choice(symbols)
            log.info(f"replay | Símbolo elegido al azar para simulación: {symbol}")

        self.symbol = symbol
        self.df: pd.DataFrame = download_bars(symbol)

        min_bars = max(config.EMA_SLOW, config.RSI_PERIOD) + 5

        # ── Filtro de fecha de inicio ──────────────────────────────────────
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date).tz_localize(ET).tz_convert("UTC")
                df_filtered = self.df[self.df.index >= start_dt].copy()
                if len(df_filtered) < min_bars:
                    last_available = self.df.index[-1].strftime("%Y-%m-%d") if len(self.df) > 0 else "N/A"
                    log.warning(
                        f"replay | {symbol} | Fecha {start_date} fuera de rango "
                        f"(último dato: {last_available}). Usando todos los datos disponibles."
                    )
                else:
                    self.df = df_filtered
                    log.info(f"replay | {symbol} | Filtro de fecha aplicado: desde {start_date}")
            except Exception as e:
                log.error(f"replay | {symbol} | Error al aplicar filtro de fecha {start_date}: {e}")

        self._index: int = 0

        if len(self.df) < min_bars:
            raise ValueError(
                f"No hay suficientes datos para {symbol}: "
                f"se necesitan {min_bars} velas, hay {len(self.df)}."
            )

        log.info(
            f"replay | {symbol} | "
            f"{len(self.df)} velas de {config.DATA_INTERVAL} disponibles | "
            f"Desde {self.df.index[0].strftime('%Y-%m-%d')} "
            f"hasta {self.df.index[-1].strftime('%Y-%m-%d')}"
        )

    def has_next(self) -> bool:
        return self._index < len(self.df)

    def next_bar(self) -> pd.Series:
        """Devuelve la siguiente vela y avanza el cursor."""
        bar = self.df.iloc[self._index]
        self._index += 1
        return bar

    def current_slice(self, window: int = 50) -> pd.DataFrame:
        """Devuelve las últimas `window` velas hasta la posición actual."""
        end   = self._index
        start = max(0, end - window)
        return self.df.iloc[start:end].copy()

    def reset(self) -> None:
        self._index = 0

    def progress_pct(self) -> float:
        return (self._index / len(self.df)) * 100


# ════════════════════════════════════════════════════════════════════════════
#  LIVE PAPER REPLAY (Modo Live Paper)
# ════════════════════════════════════════════════════════════════════════════

class LivePaperReplay:
    """
    Obtiene la vela más reciente de Alpaca API en tiempo real.

    - Modo REAL: refresca cada 30s para no exceder el rate limit de Alpaca.
    - Modo MOCK (Acelerado): recorta la historia en memoria según el reloj simulado.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        log.info(f"replay | 🚀 Iniciando Live Paper Trading para {symbol}")
        self.full_df = download_bars(symbol, force_refresh=True)
        self.df = self.full_df.copy()
        self.window = 220
        self._last_download_time = time.time()

    def has_next(self) -> bool:
        return True  # Siempre hay una vela siguiente en modo vivo

    def next_bar(self) -> pd.Series | None:
        """Devuelve la vela más reciente (real o simulada)."""
        from shared.utils.market_hours import _is_mock_time_active, now_nyc
        import pytz

        is_mock = _is_mock_time_active()

        if not is_mock:
            now = time.time()
            if (now - self._last_download_time) > 30:
                self.full_df = download_bars(self.symbol, force_refresh=True)
                self._last_download_time = now
            self.df = self.full_df.copy()
        else:
            current_mock_time = now_nyc().astimezone(pytz.UTC)
            self.df = self.full_df[self.full_df.index <= current_mock_time].copy()

        if self.df is None or self.df.empty:
            return None

        return self.df.iloc[-1]

    def current_slice(self, window: int = 50) -> pd.DataFrame:
        """Devuelve las últimas `window` velas."""
        return self.df.tail(window).copy()
