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

# ─── Zonas horarias ─────────────────────────────────────────────────────────
TZ_NYC      = pytz.timezone("America/New_York")
TZ_COLOMBIA = pytz.timezone("America/Bogota")

# ─── Horario NYSE ────────────────────────────────────────────────────────────
MARKET_OPEN  = time(9, 30)    # 9:30 AM ET
MARKET_CLOSE = time(16, 0)    # 4:00 PM ET

# Días de la semana que opera NYSE (0=Lunes … 4=Viernes)
MARKET_WEEKDAYS = set(range(5))


def now_nyc() -> datetime:
    """Devuelve la hora actual en Nueva York."""
    return datetime.now(TZ_NYC)


def now_colombia() -> datetime:
    """Devuelve la hora actual en Colombia."""
    return datetime.now(TZ_COLOMBIA)


def is_market_open() -> bool:
    """
    Retorna True si el mercado NYSE está abierto AHORA.
    Considera:
      • Día de la semana (lunes-viernes).
      • Horario 9:30-16:00 ET.
    No contempla festivos de EE.UU. (simplificación aceptable para un bot minorista).
    """
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
