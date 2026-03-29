"""
utils/market_hours.py ─ Detección del horario NYSE ajustado a Colombia.

El mercado NYSE opera de 9:30 a 16:00 ET (Eastern Time).
Colombia está en UTC-5 todo el año (sin cambio de horario).
ET puede ser UTC-5 (invierno) o UTC-4 (verano), así que la diferencia
varía entre 0 y 1 hora según el "daylight saving time" de EE.UU.

La librería pytz maneja esto automáticamente.
"""

from datetime import datetime, time, timedelta
import pytz
from shared import config

# ─── Zonas horarias ─────────────────────────────────────────────────────────
TZ_NYC      = pytz.timezone("America/New_York")
TZ_COLOMBIA = pytz.timezone("America/Bogota")

# ─── Horario NYSE ────────────────────────────────────────────────────────────
MARKET_OPEN  = time(config.TRADING_OPEN_HOUR, config.TRADING_OPEN_MIN)
MARKET_CLOSE = time(config.TRADING_CLOSE_HOUR, config.TRADING_CLOSE_MIN)

# Días de la semana que opera NYSE (0=Lunes … 4=Viernes)
MARKET_WEEKDAYS = set(range(5))


def _is_mock_time_active() -> bool:
    """Revisa si el usuario activó la simulación de las 9:30 AM para pruebas nocturnas."""
    try:
        import os, json
        from pathlib import Path
        cmd_file = config.COMMAND_FILE
        if cmd_file.exists():
            with open(cmd_file, "r") as f:
                c = json.load(f)
                return c.get("mock_time_930", False)
    except Exception:
        pass
    return False

def now_nyc() -> datetime:
    """Devuelve la hora actual en Nueva York. Si el Mock Time está activado y el mercado cerrado, simula las 9:30 AM."""
    real_nyc_now = datetime.now(TZ_NYC)
    
    # 0. Leer configuración de comandos
    cmd_config = {}
    try:
        import json
        cmd_file = config.COMMAND_FILE
        if cmd_file.exists():
            with open(cmd_file, "r") as f:
                cmd_config = json.load(f)
    except: pass
    
    is_mock_active = cmd_config.get("mock_time_930", False)
    
    # 1. Determinar si el mercado MUNDIAL REAL está abierto en este momento
    is_real_market_open = (
        real_nyc_now.weekday() in MARKET_WEEKDAYS and 
        MARKET_OPEN <= real_nyc_now.time().replace(second=0, microsecond=0) < MARKET_CLOSE
    )
    
    # 2. Si NO está abierto pero el usuario activó Test Nocturno, engañamos al bot
    if not is_real_market_open and is_mock_active:
        import time
        from shared.utils.logger import log
        
        # ── CONFIGURACIÓN DE FECHA DE REPLAY ──
        # Intentar leer fecha personalizada desde command.json
        spoof_day = real_nyc_now
        replay_date_str = cmd_config.get("sim_start_date")
        
        use_custom_time = False
        if replay_date_str:
            try:
                # Convertir "YYYY-MM-DD HH:MM" a datetime en NYC
                import pandas as pd
                spoof_day = pd.to_datetime(replay_date_str).tz_localize(TZ_NYC)
                use_custom_time = True
                # log.debug(f"MockTime | Usando fecha/hora personalizada: {replay_date_str}")
            except Exception as e:
                log.warning(f"MockTime | Error parseando sim_start_date '{replay_date_str}': {e}. Usando fallback.")
                replay_date_str = None
        
        if not replay_date_str:
            # Fallback: Encontrar el día laborable más cercano para el Mock
            if spoof_day.weekday() not in MARKET_WEEKDAYS:
                while spoof_day.weekday() not in MARKET_WEEKDAYS:
                    spoof_day -= timedelta(days=1)
                
        if use_custom_time:
            # Si el usuario puso hora (ej 11:30), la usamos tal cual
            base_mock_time = spoof_day
        else:
            # Fallback: Empezar a las 9:30:01
            base_mock_time = spoof_day.replace(
                hour=MARKET_OPEN.hour,
                minute=MARKET_OPEN.minute,
                second=1,
                microsecond=0,
            )
        
        # ── ACELERACIÓN DE TIEMPO (1s real = 1m mock) ──
        anchor_file = config.MOCK_ANCHOR_FILE
        try:
            current_real_timestamp = time.time()
            
            # Si el archivo existe, verificamos si es para la MISMA fecha de replay
            should_reset = not anchor_file.exists()
            anchor_data = {}
            if anchor_file.exists():
                try:
                    with open(anchor_file, "r") as f:
                        anchor_data = json.load(f)
                    if anchor_data.get("replay_date") != replay_date_str:
                        log.info(f"MockTime | Detectado cambio de fecha ({anchor_data.get('replay_date')} -> {replay_date_str}). Reiniciando ancla.")
                        should_reset = True
                except:
                    should_reset = True

            if should_reset:
                with open(anchor_file, "w") as f:
                    json.dump({
                        "start_time": current_real_timestamp,
                        "replay_date": replay_date_str
                    }, f)
                elapsed_seconds = 0
            else:
                elapsed_seconds = max(0.0, float(current_real_timestamp - anchor_data.get("start_time", current_real_timestamp)))
                
            # Por cada 1 segundo real que pasa, sumamos 60 segundos (1 minuto) al reloj Mock
            simulated_offset_seconds = elapsed_seconds * 60
            advanced_mock_time = base_mock_time + timedelta(seconds=simulated_offset_seconds)
            
            # Si se pasa del cierre (16:00), lo capamos en el cierre para que el bot termine el día gracefully
            close_mock_time = spoof_day.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0)
            if advanced_mock_time > close_mock_time:
                return close_mock_time
                
            return advanced_mock_time
        except Exception as e:
            log.error(f"MockTime | Error en cálculo de tiempo acelerado: {e}")
            return base_mock_time
        
    return real_nyc_now


def now_colombia() -> datetime:
    """Devuelve la hora actual en Colombia."""
    return datetime.now(TZ_COLOMBIA)


def is_market_open() -> bool:
    """
    Retorna True si el mercado NYSE está abierto AHORA (o si Test Nocturno está activo).
    Considera:
      • Día de la semana (lunes-viernes).
      • Horario 9:30-16:00 ET.
    No contempla festivos de EE.UU. (simplificación aceptable para un bot minorista).
    """
    if _is_mock_time_active():
        # En modo MOCK, dependemos exclusivamente de now_nyc() (el reloj simulado)
        # y no necesitamos el bypass anterior que era propenso a errores.
        pass
            
    nyc_now = now_nyc()
    if nyc_now.weekday() not in MARKET_WEEKDAYS:
        return False
    current_time = nyc_now.time().replace(second=0, microsecond=0)
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def time_until_open() -> timedelta | None:
    """
    Calcula cuánto tiempo falta para la próxima apertura del mercado.
    Retorna None si el mercado está abierto ahora mismo.
    """
    if is_market_open():
        return None

    nyc_now = now_nyc()
    # Calcular apertura del mismo día o del siguiente día hábil
    candidate = nyc_now.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0,
    )

    # Si ya pasó la apertura de hoy, mover al siguiente día hábil
    if nyc_now >= candidate or nyc_now.weekday() not in MARKET_WEEKDAYS:
        candidate += timedelta(days=1)
        # Saltar fin de semana
        while candidate.weekday() not in MARKET_WEEKDAYS:
            candidate += timedelta(days=1)

    return candidate - nyc_now


def next_open_str() -> str:
    """Retorna un string legible con la fecha/hora de la próxima apertura (hora de NYC y Colombia)."""
    nyc_now = now_nyc()
    candidate = nyc_now.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0,
    )
    if nyc_now >= candidate or nyc_now.weekday() not in MARKET_WEEKDAYS:
        candidate += timedelta(days=1)
        while candidate.weekday() not in MARKET_WEEKDAYS:
            candidate += timedelta(days=1)

    col_time = candidate.astimezone(TZ_COLOMBIA)
    return (
        f"{candidate.strftime('%Y-%m-%d %H:%M')} ET  "
        f"({col_time.strftime('%H:%M')} hora Colombia)"
    )


def market_status_str() -> str:
    """Devuelve un resumen del estado del mercado."""
    nyc = now_nyc()
    col = now_colombia()
    status = "🟢 ABIERTO" if is_market_open() else "🔴 CERRADO"
    return (
        f"Mercado NYSE: {status} | "
        f"NYC: {nyc.strftime('%H:%M:%S')} ET | "
        f"Colombia: {col.strftime('%H:%M:%S')}"
    )
