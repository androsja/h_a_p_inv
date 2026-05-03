"""
data/market_data.py ─ Descarga y caché de datos de mercado desde Alpaca API.

Responsabilidad única: descargar velas OHLCV, cachearlas localmente
y proveer la lista de símbolos activos.
"""

import os
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import pytz

# Alpaca imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    StockHistoricalDataClient = None

from shared import config
from shared.config import COMMAND_FILE
from shared.utils.logger import log
from shared.utils.market_hours import _is_mock_time_active, now_nyc
from shared.utils.state_writer import update_state

ET = pytz.timezone("America/New_York")


CACHE_DIR: Path = config.DATA_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 86400 * 7 # 📅 Extender vida de caché a 7 días

_assets_path = config.ASSETS_FILE

def set_assets_file(path: Path) -> None:
    global _assets_path
    _assets_path = path


def _cache_path(symbol: str) -> Path:
    interval = config.DATA_INTERVAL.replace("m","min").replace("h","hr")
    return CACHE_DIR / f"{symbol}_{interval}.parquet"


def _is_cache_valid(path: Path, required_start: Optional[datetime] = None) -> bool:
    if not path.exists():
        return False
    
    # Check age
    age = time.time() - path.stat().st_mtime
    if age > CACHE_TTL_SECONDS:
        return False

    # Check if the cache file contains the date we need
    if required_start:
        try:
            # Peak at index to check start date
            df_peak = pd.read_parquet(path, columns=[])
            if not df_peak.empty and df_peak.index[0] > required_start:
                log.info(f"cache | {path.name} | Datos inician en {df_peak.index[0]}, se requiere {required_start}. Invalidando.")
                return False
        except: return False

    return True


# ─── Filtrado de Horas ──────────────────────────────────────────────────────
def _filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
        
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert(ET)
    
    open_t  = pd.Timestamp("09:30", tz=ET).time()
    close_t = pd.Timestamp("16:00", tz=ET).time() 
    mask = (df_et.index.time >= open_t) & (df_et.index.time <= close_t)
    filtered = df_et[mask].copy()
    
    filtered.index = filtered.index.tz_convert("UTC")
    return filtered


# ─── Descarga con Alpaca API ────────────────────────────────────────────────
_alpaca_client = None

def get_alpaca_client():
    global _alpaca_client
    if _alpaca_client is None:
        if StockHistoricalDataClient is None:
            raise ImportError("La librería 'alpaca-py' no está instalada.")
        
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
             raise ValueError("Llaves de Alpaca no configuradas en el .env.")
             
        _alpaca_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY
        )
    return _alpaca_client


def download_bars(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    cache = _cache_path(symbol)
    
    # ── Calcular ventana dinámica ──
    days_to_fetch = 180
    required_start_dt = None
    
    try:
        env_start = os.environ.get("HAPI_SIM_START_DATE")
        sim_start = env_start.strip() if env_start and env_start.strip() else None
        
        env_end = os.environ.get("HAPI_SIM_END_DATE")
        sim_end = env_end.strip() if env_end and env_end.strip() else None
        
        # ... (lógica de ventana dinámica se mantiene igual) ...
        if sim_end:
            end_time_str = sim_end if ' ' in sim_end else sim_end + " 23:59:59"
            end_time = pd.to_datetime(end_time_str).tz_localize(ET).tz_convert(pytz.UTC).to_pydatetime()
            end_time = min(end_time, datetime.now(pytz.UTC))
        else:
            end_time = datetime.now(pytz.UTC)
                
        if sim_start:
            target_dt = pd.to_datetime(sim_start).tz_localize(ET).tz_convert(pytz.UTC)
            required_start_dt = target_dt - timedelta(days=7)
            diff = end_time - required_start_dt.to_pydatetime()
            days_to_fetch = max(7, diff.days + 1) if diff.days > 0 else 180
        else:
            days_to_fetch = int(config.DATA_PERIOD.replace('d', ''))
    except Exception as e:
        log.warning(f"data | Error calculando ventana dinámica: {e}")
        days_to_fetch = 180
        end_time = datetime.now(pytz.UTC)
        
    start_time = end_time - timedelta(days=days_to_fetch)

    # 🚀 1. Intentar CARGA INCREMENTAL (Solo descargar lo que falta)
    df_existing = None
    if not force_refresh and cache.exists():
        try:
            df_existing = pd.read_parquet(cache)
            if not df_existing.empty:
                # Si el caché tiene datos que cubren el inicio requerido, solo necesitamos el final
                last_ts = df_existing.index[-1]
                # Si el último dato del caché es de hace menos de 5 minutos (Live) o es posterior al end_time (Sim)
                if last_ts >= (end_time - timedelta(minutes=5)):
                    log.info(f"data | {symbol} | Caché completa. Cargando {len(df_existing)} velas.")
                    return df_existing
                
                # Caso incremental: El caché existe pero faltan velas recientes
                log.info(f"data | {symbol} | Caché detectada. Realizando descarga incremental desde {last_ts}")
                start_time = last_ts + timedelta(minutes=1)
        except Exception as e:
            log.warning(f"data | Error leyendo caché para incremental: {e}")
            df_existing = None

    # 2. Descargar de Alpaca (sea el bloque completo o solo la parte incremental)
    try:
        log.info(f"data | {symbol} | Solicitando datos a Alpaca desde {start_time.strftime('%Y-%m-%d %H:%M')}")
        
        # ── Progreso UI ──
        update_state(symbol=symbol, status="downloading", download_progress=10, download_symbol=symbol)
        
        client = get_alpaca_client()
        val = int(config.DATA_INTERVAL.replace("m", ""))
        tf = TimeFrame(val, TimeFrameUnit.Minute)
        
        # Fragmentar en chunks si la ventana es muy grande (> 60 días)
        all_chunks = []
        cur_start = start_time
        chunk_days = 60
        
        while cur_start < end_time:
            cur_end = min(cur_start + timedelta(days=chunk_days), end_time)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf,
                start=cur_start, end=cur_end, feed="iex"
            )
            bars = client.get_stock_bars(request_params)
            if not bars.df.empty:
                raw_chunk = bars.df.xs(symbol) if symbol in bars.df.index.levels[0] else bars.df
                all_chunks.append(raw_chunk)
            cur_start = cur_end
            
        if not all_chunks:
            if df_existing is not None: return df_existing
            raise ValueError(f"No hay datos nuevos para {symbol}.")
            
        new_df = pd.concat(all_chunks)
        new_df = new_df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
        new_df.index = pd.to_datetime(new_df.index, utc=True)
        new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].copy()
        new_df.dropna(inplace=True)
        new_df = _filter_trading_hours(new_df)

        # 🧩 Unir (Merge) con caché previo si existe
        if df_existing is not None:
            final_df = pd.concat([df_existing, new_df])
            # Eliminar duplicados y ordenar por tiempo
            final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
            # Mantener solo la ventana de días requerida para no crecer infinitamente
            cutoff = end_time - timedelta(days=days_to_fetch + 10)
            final_df = final_df[final_df.index >= cutoff]
            df = final_df
        else:
            df = new_df

        # Guardar en disco
        df.to_parquet(cache)
        log.info(f"data | {symbol} | Descarga finalizada y caché saneada ({len(df)} velas).")
        return df
        
    except Exception as exc:
        if df_existing is not None:
            log.warning(f"data | {symbol} | Falló descarga incremental, usando caché previa. Error: {exc}")
            return df_existing
        log.error(f"data | {symbol} | Error Alpaca: {exc}")
        raise

    # 2. (El Recorte Dinámico fue eliminado. LivePaperReplay y MarketReplay ya manejan el flujo temporal independientemente).
    return df


def get_symbols() -> list[str]:
    try:
        path = _assets_path if _assets_path.exists() else config.ASSETS_FILE
        if not path.exists(): return ["AAPL"]
        with open(path, "r") as f:
            data = json.load(f)
            assets_list = data.get("assets", [])
            active = [a["symbol"] for a in assets_list if a.get("enabled", True)]
            return active if active else ["AAPL"]
    except Exception as exc:
        log.error(f"data | Error leyendo assets: {exc}")
        return ["AAPL"]

# Alias
download_1m = download_bars

def download_all(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for symbol in get_symbols():
        try:
            result[symbol] = download_bars(symbol, force_refresh=force_refresh)
        except Exception as exc:
            log.error(f"data | {symbol} | Omitiendo: {exc}")
    return result

__all__ = ["download_bars", "download_1m", "download_all", "get_symbols", "set_assets_file"]
