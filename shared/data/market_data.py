"""
data/market_data.py ─ Descarga y caché de datos de mercado desde Alpaca API.

Responsabilidad única: descargar velas OHLCV, cachearlas localmente
y proveer la lista de símbolos activos.

Los reproductores de datos (MarketReplay, LivePaperReplay) viven en replay.py.
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
            
            # Calcular ventana de tiempo dinámica basada en DATA_PERIOD
            try:
                days_to_fetch = int(config.DATA_PERIOD.replace('d', ''))
            except:
                days_to_fetch = 180
                
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(days=days_to_fetch)
            
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
            # Truncar el mensaje (puede ser HTML de error 500 de nginx/Alpaca)
            exc_msg = str(exc).replace("\n", " ").replace("\r", " ").strip()[:120]
            log.error(f"data | {symbol} | Error descargando desde Alpaca: {exc_msg}")
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


# ─── Re-exports de compatibilidad eliminados para evitar importación circular ─────
# Si se requieren los reproductores, importarlos directamente desde shared.data.replay

__all__ = [
    "download_bars", "download_1m", "download_all",
    "get_symbols", "set_assets_file",
]
