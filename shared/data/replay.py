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
from typing import Optional, Union

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

    def __init__(self, symbol: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Args:
            symbol:     Ticker a simular. Si es None, elige uno al azar de assets.json.
            start_date: Formato "YYYY-MM-DD". Filtra el inicio de la simulación.
            end_date:   Formato "YYYY-MM-DD". Filtra el fin de la simulación.
        """
        # ── Normalizar fechas: string vacío → None ─────────────────────────
        # Evita que end_date="" (cadena vacía desde env var) bypass el filtro
        if not start_date or not start_date.strip():
            start_date = None
        if not end_date or not end_date.strip():
            end_date = None

        log.info(f"replay | {symbol or 'AUTO'} | Fechas recibidas → inicio: {start_date or 'no definida'} | fin: {end_date or 'no definida'}")

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
                        f"replay | {symbol} | Fecha inicio {start_date} fuera de rango "
                        f"(último dato: {last_available}). Usando todos los datos disponibles."
                    )
                else:
                    self.df = df_filtered
                    log.info(f"replay | {symbol} | ✅ Filtro de INICIO aplicado: desde {start_date}")
            except Exception as e:
                log.error(f"replay | {symbol} | Error al aplicar filtro de fecha inicio {start_date}: {e}")

        # ── Filtro de fecha de fin ─────────────────────────────────────────
        if end_date:
            try:
                # Fin de día para incluir toda la jornada del end_date
                end_dt_str = end_date if ' ' in end_date else end_date + " 23:59:59"
                end_dt = pd.to_datetime(end_dt_str).tz_localize(ET).tz_convert("UTC")
                df_filtered = self.df[self.df.index <= end_dt].copy()
                if len(df_filtered) < min_bars:
                    log.warning(
                        f"replay | {symbol} | ⚠️ Fecha fin {end_date} deja solo {len(df_filtered)} velas "
                        f"(mínimo {min_bars}). Ignorando filtro fin para evitar crash."
                    )
                else:
                    self.df = df_filtered
                    log.info(f"replay | {symbol} | ✅ Filtro de FIN aplicado: hasta {end_date} ({len(df_filtered)} velas)")
            except Exception as e:
                log.error(f"replay | {symbol} | Error al aplicar filtro de fecha fin {end_date}: {e}")
        else:
            log.info(f"replay | {symbol} | ℹ️  Sin fecha de fin definida → simulará hasta la última vela disponible ({self.df.index[-1].strftime('%Y-%m-%d') if len(self.df) > 0 else 'N/A'})")

        self._index: int = 0

        # ── Log confirmación del rango real que se va a simular ────────────
        if len(self.df) > 0:
            log.info(
                f"replay | {symbol} | 🗓️  RANGO EFECTIVO DE SIMULACIÓN: "
                f"{self.df.index[0].strftime('%Y-%m-%d')} → {self.df.index[-1].strftime('%Y-%m-%d')} "
                f"({len(self.df)} velas)"
            )

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

    def peek_next_timestamp(self) -> Optional[pd.Timestamp]:
        """Devuelve el timestamp de la siguiente vela sin avanzar el cursor."""
        if not self.has_next():
            return None
        return self.df.index[self._index]

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
        self._warned_no_data = False

    def has_next(self) -> bool:
        """En modo Mock Time, termina la sesión si llegamos a las 4:00 PM NYC."""
        from shared.utils.market_hours import _is_mock_time_active, now_nyc
        from datetime import time as dt_time
        
        if _is_mock_time_active():
            if now_nyc().time() >= dt_time(16, 0):
                return False
        return True

    def next_bar(self) -> Optional[pd.Series]:
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
            
            # Verificación de datos disponibles
            if not self.full_df.empty and current_mock_time > self.full_df.index[-1]:
                if not getattr(self, "_warned_no_data", False):
                    log.warning(f"replay | {self.symbol} | El reloj simulado ({current_mock_time}) superó los datos disponibles.")
                    self._warned_no_data = True
            
            self.df = self.full_df[self.full_df.index <= current_mock_time].copy()

        return self.df.tail(window).copy()
