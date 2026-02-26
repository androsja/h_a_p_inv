"""
config.py ─ Configuración central del sistema de trading
Carga variables del .env y provee constantes globales.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Cargar .env ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ─── Credenciales Hapi ──────────────────────────────────────────────────────
HAPI_API_KEY    = os.getenv("HAPI_API_KEY", "")
HAPI_CLIENT_ID  = os.getenv("HAPI_CLIENT_ID", "")
HAPI_USER_TOKEN = os.getenv("HAPI_USER_TOKEN", "")

# URL base de la API de Hapi Trade
HAPI_BASE_URL   = "https://api.hapitrade.com/v1"

# ─── Modo de operación ──────────────────────────────────────────────────────
TRADING_MODE    = os.getenv("TRADING_MODE", "SIMULATED").upper()   # LIVE | SIMULATED
HAPI_TEST_MODE  = os.getenv("HAPI_IS_TEST_MODE", "true").lower() == "true"

# ─── Gestión de riesgo (Intraday) ───────────────────────────────────────────
STOP_LOSS_PCT       = float(os.getenv("STOP_LOSS_PCT",    "0.025"))   # 2.5% SL
TAKE_PROFIT_PCT     = float(os.getenv("TAKE_PROFIT_PCT",  "0.05"))    # 5.0% TP  → ratio 1:2
MAX_POSITION_USD    = float(os.getenv("MAX_POSITION_USD", "10000"))   # Relajado a tamaño total de la cuenta
MAX_RISK_PCT        = float(os.getenv("MAX_RISK_PCT", "0.01"))      # Máximo riesgo de 1% del capital por trade
CLEARING_COST_USD   = float(os.getenv("CLEARING_COST_USD","0.15"))   # $0.15 Apex

# ─── Latencia Bogotá → Nueva York ───────────────────────────────────────────
LATENCY_OFFSET_CENTS = float(os.getenv("LATENCY_OFFSET_CENTS", "2")) / 100

# ─── Logs ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = BASE_DIR / os.getenv("LOG_FILE", "logs/trading_bot.log")

# ─── Estrategia Intraday (velas de 5 minutos) ────────────────────────────────
# EMA 12/26 es el estándar para intraday (mismos períodos que el MACD clásico)
EMA_FAST       = 12     # EMA rápida  — señal de entrada
EMA_SLOW       = 26     # EMA lenta   — tendencia de fondo
RSI_PERIOD     = 14     # RSI estándar
RSI_OVERBOUGHT = 70     # Más estricto: no comprar si RSI > 70 (evita sobrecompra)
RSI_OVERSOLD   = 35     # Más estricto: señal de impulso si RSI < 35

# ─── Datos de mercado ────────────────────────────────────────────────────────
# Intervalo y período para yfinance (intraday en lugar de scalping)
DATA_INTERVAL  = "5m"   # Velas de 5 minutos  (yfinance permite 5m × 60d gratis)
DATA_PERIOD    = "60d"  # 60 días de historial → ~7,800 velas por símbolo

# ─── Ventana de trading: apertura de NY (mayor liquidez intraday) ────────────
# Solo operamos de 9:30 a 11:30 AM ET (primeras 2 horas = 80% del volumen diario)
TRADING_OPEN_HOUR   = 9    # Hora de apertura (NY)
TRADING_OPEN_MIN    = 30
TRADING_CLOSE_HOUR  = 11   # Hora de cierre de ventana
TRADING_CLOSE_MIN   = 30

# ─── Archivos de datos ──────────────────────────────────────────────────────
ASSETS_FILE      = BASE_DIR / "assets.json"
DATA_CACHE_DIR   = BASE_DIR / "data" / "cache"

# ─── Timing del bot ─────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC  = 1    # Evaluar cada 1 segundo en LIVE (entre velas de 5min)
REST_INTERVAL_SEC  = 60   # Descanso fuera de ventana de trading

# ─── Settlement T+1 ─────────────────────────────────────────────────────────
SETTLEMENT_DAYS = 1
