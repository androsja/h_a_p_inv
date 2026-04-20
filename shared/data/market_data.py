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


# ─── Caché ──────────────────────────────────────────────────────────────────
CACHE_DIR: Path = config.DATA_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 86400   # Refrescar caché cada 24 horas (más eficiente para simulación autónoma)

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
        
        if not sim_start and COMMAND_FILE.exists():
            with open(COMMAND_FILE, "r") as f:
                cmds = json.load(f)
                sim_start = cmds.get("sim_start_date")
                if not sim_end:
                    sim_end = cmds.get("sim_end_date")
                    
        if sim_end:
            end_time_str = sim_end if ' ' in sim_end else sim_end + " 23:59:59"
            end_time = pd.to_datetime(end_time_str).tz_localize(ET).tz_convert(pytz.UTC).to_pydatetime()
            end_time = min(end_time, datetime.now(pytz.UTC)) # No futuro
        else:
            end_time = datetime.now(pytz.UTC)
                
        if sim_start:
            target_dt = pd.to_datetime(sim_start).tz_localize(ET).tz_convert(pytz.UTC)
            required_start_dt = target_dt - timedelta(days=7) # Buffer para indicadores
            diff = end_time - required_start_dt.to_pydatetime()
            # Si target_dt está en el futuro respecto a end_time, no intentes descargar un rango irreal
            days_to_fetch = max(7, diff.days + 1) if diff.days > 0 else 180
            log.info(f"data | {symbol} | Ventana dinámica calculada: {days_to_fetch} días (sim_start: {sim_start}, sim_end: {sim_end})")
        else:
            days_to_fetch = int(config.DATA_PERIOD.replace('d', ''))
    except Exception as e:
        log.warning(f"data | Error calculando ventana dinámica: {e}")
        days_to_fetch = 180
        end_time = datetime.now(pytz.UTC)
        
    start_time = end_time - timedelta(days=days_to_fetch)

    # 1. Intentar cargar desde Caché
    df = None
    if not force_refresh and _is_cache_valid(cache, required_start=required_start_dt):
        try:
            df = pd.read_parquet(cache)
        except: pass

    if df is None or df.empty:
        log.info(f"data | {symbol} | Descargando desde Alpaca API...")
        try:
            # ── Calcular número total de chunks para la barra de progreso ──
            chunk_days = 60
            _total_chunks = max(1, int((days_to_fetch + chunk_days - 1) / chunk_days))
            update_state(symbol=symbol, status="downloading",
                         download_progress=0, download_symbol=symbol,
                         download_total_chunks=_total_chunks, download_current_chunk=0)
            client = get_alpaca_client()
            val = int(config.DATA_INTERVAL.replace("m", ""))
            tf = TimeFrame(val, TimeFrameUnit.Minute)
            
            # FETCH POR CHUNKS DE 60 DÍAS PARA EVITAR LIMITES DE LA API DE ALPACA (10,000 ROWS MAX)
            all_chunks = []
            cur_start = start_time
            _chunk_idx = 0
            
            while cur_start < end_time:
                cur_end = min(cur_start + timedelta(days=chunk_days), end_time)
                _chunk_idx += 1
                _pct = min(99, int((_chunk_idx / _total_chunks) * 100))
                chunk_label = cur_start.strftime("%Y-%m-%d")
                update_state(symbol=symbol, status="downloading",
                             download_progress=_pct, download_symbol=symbol,
                             download_total_chunks=_total_chunks,
                             download_current_chunk=_chunk_idx,
                             download_chunk_date=chunk_label)
                log.info(f"data | {symbol} | Chunk {_chunk_idx}/{_total_chunks} ({_pct}%) desde {chunk_label}")
                
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=cur_start,
                    end=cur_end,
                    feed="iex"
                )
                
                try:
                    bars = client.get_stock_bars(request_params)
                    if not bars.df.empty:
                        raw_chunk = bars.df.xs(symbol) if symbol in bars.df.index.levels[0] else bars.df
                        all_chunks.append(raw_chunk)
                except Exception as e:
                    log.warning(f"data | partial error for chunk {cur_start}: {e}")
                
                cur_start = cur_end
                
            if not all_chunks:
                raise ValueError(f"No hay datos para {symbol} en Alpaca en este rango.")
                
            raw_df = pd.concat(all_chunks)
            # Evitar superposiciones en caso de solapamiento de ventanas
            raw_df = raw_df[~raw_df.index.duplicated(keep='first')]
            
            raw_df = raw_df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"
            })
            raw_df.index = pd.to_datetime(raw_df.index, utc=True)
            raw_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
            raw_df.dropna(inplace=True)
            
            df = _filter_trading_hours(raw_df)
            df.to_parquet(cache)
            log.info(f"data | {symbol} | Caché actualizada ({len(df)} velas de {days_to_fetch} días).")
            
        except Exception as exc:
            exc_msg = str(exc).replace("\n", " ").strip()[:120]
            log.error(f"data | {symbol} | Error Alpaca: {exc_msg}")
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
