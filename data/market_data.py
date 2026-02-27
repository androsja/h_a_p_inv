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
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytz
import yfinance as yf

import config
from utils.logger import log

ET = pytz.timezone("America/New_York")


# ─── Caché ──────────────────────────────────────────────────────────────────
CACHE_DIR: Path = config.DATA_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 3600   # Refrescar caché cada hora (las velas de 5m son más estables)


def _cache_path(symbol: str) -> Path:
    interval = config.DATA_INTERVAL.replace("m","min").replace("h","hr")
    return CACHE_DIR / f"{symbol}_{interval}.parquet"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_TTL_SECONDS


# ─── Carga de activos ────────────────────────────────────────────────────────
def load_assets() -> list[dict]:
    """Lee el archivo assets.json y retorna la lista de activos."""
    with open(config.ASSETS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Filtrar solo los activos que están habilitados (o los que no tengan el flag, por defecto True)
    return [a for a in data["assets"] if a.get("enabled", True)]


def get_symbols() -> list[str]:
    """Retorna solo la lista de símbolos del archivo assets.json."""
    return [a["symbol"] for a in load_assets()]


# ─── Descarga con yfinance ───────────────────────────────────────────────────
def _filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el DataFrame para conservar solo velas dentro de la
    ventana de trading intraday: 9:30 – 11:30 AM ET.
    El 80% del volumen diario ocurre en las primeras 2 horas.
    """
    if df.empty:
        return df
    # Convertir índice a ET para poder filtrar por hora
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert(ET)
    open_t  = pd.Timestamp("09:30", tz=ET).time()
    close_t = pd.Timestamp("11:30", tz=ET).time()
    mask = (df_et.index.time >= open_t) & (df_et.index.time <= close_t)
    filtered = df_et[mask].copy()
    # Volver a UTC para consistencia interna
    filtered.index = filtered.index.tz_convert("UTC")
    return filtered


def download_bars(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Descarga datos OHLCV en el intervalo configurado (5 minutos, 60 días).
    Filtra al horario de apertura NYSE 9:30–11:30 AM ET.
    Usa caché local para evitar llamadas repetidas a Yahoo Finance.
    """
    cache = _cache_path(symbol)

    if not force_refresh and _is_cache_valid(cache):
        log.debug(f"data | {symbol} | Cargando desde caché: {cache.name}")
        return pd.read_parquet(cache)

    log.info(
        f"data | {symbol} | Descargando {config.DATA_INTERVAL} × "
        f"{config.DATA_PERIOD} desde yfinance…"
    )
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=config.DATA_PERIOD,
            interval=config.DATA_INTERVAL,
            auto_adjust=True,
        )

        if df.empty:
            log.error(f"data | {symbol} | yfinance devolvió un DataFrame vacío.")
            if cache.exists():
                log.warning(f"data | {symbol} | Usando caché antiguo como respaldo.")
                return pd.read_parquet(cache)
            raise ValueError(f"No hay datos disponibles para {symbol}")

        # Normalizar y filtrar
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df = _filter_trading_hours(df)   # Solo 9:30–11:30 AM ET
        
        # Test Nocturno (Mock Time): Recortar el DataFrame para no "ver el futuro"
        from utils.market_hours import now_nyc
        current_mock_time = now_nyc().astimezone(pytz.UTC)
        df = df[df.index <= current_mock_time].copy()

        # Guardar caché
        df.to_parquet(cache)
        log.info(
            f"data | {symbol} | {len(df)} velas de {config.DATA_INTERVAL} "
            f"(horario 9:30-11:30 ET) descargadas y en caché."
        )
        return df

    except Exception as exc:
        log.error(f"data | {symbol} | Error descargando datos: {exc}")
        if cache.exists():
            log.warning(f"data | {symbol} | Fallback a caché existente.")
            return pd.read_parquet(cache)
        raise


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

    def __init__(self, symbol: str | None = None):
        """
        Si symbol es None, elige uno al azar de assets.json.
        """
        if symbol is None:
            symbols = get_symbols()
            symbol = random.choice(symbols)
            log.info(f"replay | Símbolo elegido al azar para simulación: {symbol}")

        self.symbol = symbol
        self.df: pd.DataFrame = download_bars(symbol)   # 5min × 60d
        self._index: int = 0

        min_bars = max(config.EMA_SLOW, config.RSI_PERIOD) + 5
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
    Simula datos reales consultando yfinance cada 60 segundos en tiempo real.
    Mantiene la historia para los indicadores usando una ventana móvil.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        log.info(f"replay | 🚀 Iniciando Live Paper Trading (Broker Sombra) para {symbol}")
        # Descarga historia base para contexto
        self.df = download_bars(symbol, force_refresh=True)
        self.window = 220
        
    def has_next(self) -> bool:
        return True # Siempre hay una siguiente vela en modo vivo
        
    def next_bar(self) -> pd.Series:
        """Descarga la última vela esperando si es necesario"""
        import time
        # En vez de bloquear aquí 60s, fetch data y return the last row
        self.df = download_bars(self.symbol, force_refresh=True)
        return self.df.iloc[-1]
        
    def current_slice(self, window: int = 50) -> pd.DataFrame:
        """Devuelve las últimas `window` velas."""
        return self.df.tail(window).copy()
