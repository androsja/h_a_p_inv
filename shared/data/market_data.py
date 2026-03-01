"""
data/market_data.py ─ Descarga y caché de datos de mercado con yfinance.

Estrategia intraday: velas de 5 minutos, últimos 60 días (gratis con yfinance).
Filtradas al horario de alta liquidez: 9:30–11:30 AM ET (apertura de NYSE).

En modo SIMULATED el MockBroker usa estos datos para "reproducir" el mercado
como si fuera en vivo, vela a vela.
"""

import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytz

# Alpaca imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    # Fallback si todavía no se ha reconstruido el contenedor
    StockHistoricalDataClient = None

from shared import config
from shared.utils.logger import log

ET = pytz.timezone("America/New_York")


# ─── Caché ──────────────────────────────────────────────────────────────────
CACHE_DIR: Path = config.DATA_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 3600   # Refrescar caché cada hora (las velas de 5m son más estables)

_assets_path = config.ASSETS_FILE

def set_assets_file(path: Path) -> None:
    global _assets_path
    _assets_path = path


def _cache_path(symbol: str) -> Path:
    interval = config.DATA_INTERVAL.replace("m","min").replace("h","hr")
    return CACHE_DIR / f"{symbol}_{interval}.parquet"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    
    age = time.time() - path.stat().st_mtime
    
    from shared.utils.market_hours import _is_mock_time_active
    if _is_mock_time_active():
        return age < CACHE_TTL_SECONDS
        
    return age < CACHE_TTL_SECONDS


# ─── Filtrado de Horas ──────────────────────────────────────────────────────
def _filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame para conservar solo velas dentro de la
    ventana de trading intraday: 9:30 – 16:00 ET.
    """
    if df.empty:
        return df
    # Asegurar que el índice tenga zona horaria (y que sea UTC para convertir a ET)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert(ET)
    
    open_t  = pd.Timestamp("09:30", tz=ET).time()
    close_t = pd.Timestamp("16:00", tz=ET).time() 
    mask = (df_et.index.time >= open_t) & (df_et.index.time <= close_t)
    filtered = df_et[mask].copy()
    
    # Volver a UTC para consistencia interna
    filtered.index = filtered.index.tz_convert("UTC")
    return filtered


# ─── Descarga con Alpaca API ────────────────────────────────────────────────
_alpaca_client = None

def get_alpaca_client():
    global _alpaca_client
    if _alpaca_client is None:
        if StockHistoricalDataClient is None:
            raise ImportError("La librería 'alpaca-py' no está instalada. Ejecute 'pip install alpaca-py'.")
        
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
             raise ValueError("Las llaves de Alpaca (APCA_API_KEY_ID/SECRET) no están configuradas en el .env.")
             
        _alpaca_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY
        )
    return _alpaca_client


def download_bars(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Descarga datos OHLCV usando Alpaca API.
    Filtra al horario de apertura NYSE 9:30–16:00 ET.
    Usa caché local Parquet para evitar excesivas peticiones.
    """
    cache = _cache_path(symbol)
    from shared.utils.market_hours import _is_mock_time_active, now_nyc

    # 1. Intentar cargar desde Caché
    df = None
    if not force_refresh and _is_cache_valid(cache):
        try:
            df = pd.read_parquet(cache)
        except Exception:
            pass

    if df is None or df.empty:
        log.info(f"data | {symbol} | Descargando {config.DATA_INTERVAL} desde Alpaca API...")
        try:
            from shared.utils.state_writer import update_state
            update_state(symbol=symbol, status="downloading")
            client = get_alpaca_client()
            
            # Calcular ventana de tiempo (60 días atrás)
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(days=60)
            
            # Configurar timeframe (5m -> TimeFrame(5, TimeFrameUnit.Minute))
            val = int(config.DATA_INTERVAL.replace("m", ""))
            tf = TimeFrame(val, TimeFrameUnit.Minute)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_time,
                end=end_time,
                feed="iex"  # "iex" es gratis, "sip" requiere suscripción
            )
            
            bars = client.get_stock_bars(request_params)
            
            if bars.df.empty:
                log.error(f"data | {symbol} | Alpaca devolvió un DataFrame vacío.")
                raise ValueError(f"No hay datos disponibles para {symbol} en Alpaca.")

            # Alpaca devuelve MultiIndex (symbol, timestamp). Aplanamos.
            raw_df = bars.df.xs(symbol) if symbol in bars.df.index.levels[0] else bars.df
            
            # Normalizar columnas a los nombres esperados por el bot (Open, High, Low, Close, Volume)
            raw_df = raw_df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            
            # Asegurar que el índice es datetime y UTC
            raw_df.index = pd.to_datetime(raw_df.index, utc=True)
            raw_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
            raw_df.dropna(inplace=True)
            
            # Filtrar y guardar
            df = _filter_trading_hours(raw_df)
            df.to_parquet(cache)
            log.info(f"data | {symbol} | Caché Alpaca actualizada ({len(df)} velas).")
            
        except Exception as exc:
            log.error(f"data | {symbol} | Error descargando desde Alpaca: {exc}")
            raise
            
    # 2. Recorte Dinámico (Modo Mock Time)
    if _is_mock_time_active():
        current_mock_time = now_nyc().astimezone(pytz.UTC)
        df = df[df.index <= current_mock_time].copy()
        
    return df


def get_symbols() -> list[str]:
    """Lee la lista de símbolos activos desde el archivo configurado."""
    try:
        if not _assets_path.exists():
            # Si el específico no existe, probar el genérico
            if config.ASSETS_FILE.exists():
                path = config.ASSETS_FILE
            else:
                return ["AAPL"]
        else:
            path = _assets_path

        with open(path, "r") as f:
            data = json.load(f)
            assets_list = data.get("assets", [])
            enabled_symbols = [a["symbol"] for a in assets_list if a.get("enabled", True)]
            log.info(f"data | 🔄 Catálogo cargado ({path.name}): {len(enabled_symbols)} símbolos activos.")
            return enabled_symbols
    except Exception as exc:
        log.error(f"data | ❌ Error crítico leyendo assets.json: {exc}")
        return ["AAPL"]


# Alias para compatibilidad con imports existentes
download_1m = download_bars


def download_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Descarga datos para todos los activos en assets.json."""
    result: dict[str, pd.DataFrame] = {}
    for symbol in get_symbols():
        try:
            result[symbol] = download_bars(symbol, force_refresh=force_refresh)
        except Exception as exc:
            log.error(f"data | {symbol} | Omitiendo por error: {exc}")
    return result


# ─── Reproductor de datos para modo simulado ────────────────────────────────
class MarketReplay:
    """
    Reproduce los datos históricos de 5 minutos (ventana 9:30–11:30 ET)
    como si fueran datos en vivo vela a vela.
    El MockBroker usa esta clase para simular el mercado intraday.
    """

    def __init__(self, symbol: str | None = None, start_date: str | None = None):
        """
        Si symbol es None, elige uno al azar de assets.json.
        start_date: Formato "YYYY-MM-DD" para filtrar el inicio de la simulación.
        """
        if symbol is None:
            symbols = get_symbols()
            symbol = random.choice(symbols)
            log.info(f"replay | Símbolo elegido al azar para simulación: {symbol}")

        self.symbol = symbol
        self.df: pd.DataFrame = download_bars(symbol)   # 5min × 60d

        # Aplicar filtro de fecha de inicio si se proporciona
        min_bars = max(config.EMA_SLOW, config.RSI_PERIOD) + 5
        if start_date:
            try:
                # Convertir start_date a datetime UTC para comparar con el índice
                start_dt = pd.to_datetime(start_date).tz_localize(ET).tz_convert("UTC")
                df_filtered = self.df[self.df.index >= start_dt].copy()
                # Si la fecha está en el futuro o no hay datos suficientes, usar todos los datos
                if len(df_filtered) < min_bars:
                    last_available = self.df.index[-1].strftime("%Y-%m-%d") if len(self.df) > 0 else "N/A"
                    log.warning(
                        f"replay | {symbol} | Fecha {start_date} fuera de rango o sin datos suficientes "
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
            f"hasta {self.df.index[-1].strftime('%Y-%m-%d')} | "
            f"Horario: 9:30–11:30 AM ET"
        )

    def has_next(self) -> bool:
        return self._index < len(self.df)

    def next_bar(self) -> pd.Series:
        """Devuelve la siguiente vela y avanza el cursor."""
        bar = self.df.iloc[self._index]
        self._index += 1
        return bar

    def current_slice(self, window: int = 50) -> pd.DataFrame:
        """
        Devuelve las últimas `window` velas hasta la posición actual.
        Necesario para calcular indicadores técnicos.
        """
        end = self._index
        start = max(0, end - window)
        return self.df.iloc[start:end].copy()

    def reset(self) -> None:
        self._index = 0

    def progress_pct(self) -> float:
        return (self._index / len(self.df)) * 100

class LivePaperReplay:
    """
    Simula datos reales consultando Alpaca API.
    - En modo REAL: Consulta cada 60s (con throttle).
    - En modo MOCK (Acelerado): Replay de la historia cacheada en memoria.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        log.info(f"replay | 🚀 Iniciando Live Paper Trading (Broker Sombra) para {symbol}")
        # Descarga historia base para contexto inicial y la mantenemos en memoria para velocidad
        self.full_df = download_bars(symbol, force_refresh=True)
        self.df = self.full_df.copy() # Slice actual
        self.window = 220
        self._last_download_time = time.time()
        
    def has_next(self) -> bool:
        return True # Siempre hay una siguiente vela en modo vivo
        
    def next_bar(self) -> pd.Series | None:
        """Devuelve la vela actual (real o simulada)"""
        from shared.utils.market_hours import _is_mock_time_active, now_nyc
        import pytz
        
        is_mock = _is_mock_time_active()
        
        if not is_mock:
            now = time.time()
            # Refrescar solo cada 30s en modo real para ser amables con Alpaca
            if (now - self._last_download_time) > 30:
                self.full_df = download_bars(self.symbol, force_refresh=True)
                self._last_download_time = now
            self.df = self.full_df.copy()
        else:
            # En modo MOCK (Acelerado), recortamos la historia en memoria según el reloj simulado
            current_mock_time = now_nyc().astimezone(pytz.UTC)
            self.df = self.full_df[self.full_df.index <= current_mock_time].copy()
            
        if self.df is None or self.df.empty:
            return None
            
        return self.df.iloc[-1]
        
    def current_slice(self, window: int = 50) -> pd.DataFrame:
        """Devuelve las últimas `window` velas."""
        return self.df.tail(window).copy()
