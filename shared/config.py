"""
config.py ─ Configuración central del sistema de trading
Carga variables del .env y provee constantes globales.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    BASE_DIR = Path(__file__).resolve().parent
    if BASE_DIR.name == "shared":
        BASE_DIR = BASE_DIR.parent
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    # Si no está instalado, simplemente ignoramos. El bot usará variables de entorno de shell si hay.
    BASE_DIR = Path(__file__).resolve().parent
    if BASE_DIR.name == "shared":
        BASE_DIR = BASE_DIR.parent

# URL base de la API de Hapi Trade
HAPI_BASE_URL     = os.getenv("HAPI_BASE_URL", "https://api.hapitrade.com/v1")
HAPI_API_KEY      = os.getenv("HAPI_API_KEY", "")
HAPI_API_SECRET   = os.getenv("HAPI_API_SECRET", "")

# ─── Credenciales Alpaca (para datos de mercado) ──────────────────────────
ALPACA_API_KEY      = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY   = os.getenv("APCA_API_SECRET_KEY", "")
ALPACA_BASE_URL     = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

# ─── Modo de operación ──────────────────────────────────────────────────────
TRADING_MODE    = os.getenv("TRADING_MODE", "SIMULATED").upper()   # LIVE | SIMULATED
HAPI_TEST_MODE  = os.getenv("HAPI_IS_TEST_MODE", "true").lower() == "true"

# ── RISK MANAGEMENT (Configured for IBKR) ───────────────────────────
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT", "0.01"))      # 1% por trade
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT", "0.02"))    # 2% por trade
MAX_POSITION_USD  = float(os.getenv("MAX_POSITION_USD", "5000.0")) # Tamaño máximo de posición en vivo
INITIAL_CASH_LIVE = float(os.getenv("INITIAL_CASH_LIVE", "10000.0")) # Balance inicial forzado para Live Paper

# ─── Latencia Bogotá → Nueva York ───────────────────────────────────────────
LATENCY_OFFSET_CENTS = float(os.getenv("LATENCY_OFFSET_CENTS", "2")) / 100

# ─── Logs ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = BASE_DIR / os.getenv("LOG_FILE", "logs/trading_bot.log")

# ─── Estrategia Intraday (velas de 5 minutos) ────────────────────────────────
# EMA 12/26 es el estándar para intraday (mismos períodos que el MACD clásico)
EMA_FAST       = 12     # EMA rápida  — señal de entrada
# Aumentamos exigencia para reducir trades y comisiones (antes 0.45)
CONFIDENCE_THRESHOLD = 0.60
EMA_SLOW       = 26     # EMA lenta   — tendencia de fondo
RSI_PERIOD     = 14     # RSI estándar
RSI_OVERBOUGHT = 75     # Más agresivo: permite tendencias fuertes (antes 70)
RSI_OVERSOLD   = 40     # Más agresivo: entra más temprano en rebotes (antes 35)

# ─── Datos de mercado ────────────────────────────────────────────────────────
# Intervalo y período para yfinance (intraday en lugar de scalping)
DATA_INTERVAL  = "5m"   # Regresando a 5m para mayor estabilidad, manteniendo 20 activos.
DATA_PERIOD    = "180d" # 180 días de historial (6 meses) para entrenar a la IA a profundidad

# ─── Ventana de trading: apertura de NY (mayor liquidez intraday) ────────────
# 🕒 Ventana de Trading (Horario NY)
# Hiper-Agresivo V3: Expandimos la ventana de 2 a 4 horas.
TRADING_OPEN_HOUR   = 9    # Hora de apertura (NY)
TRADING_OPEN_MIN    = 30
TRADING_CLOSE_HOUR  = 16   # Hora de cierre de ventana (4:00 PM ET)
TRADING_CLOSE_MIN   = 0

# ─── Archivos de datos ──────────────────────────────────────────────────────
ASSETS_FILE      = BASE_DIR / "assets.json"
ASSETS_FILE_SIM  = ASSETS_FILE
ASSETS_FILE_LIVE = ASSETS_FILE
DATA_DIR         = BASE_DIR / "data"
COMMAND_FILE     = DATA_DIR / "command.json"
STATE_FILE       = DATA_DIR / "state.json" # Legacy/Default
STATE_FILE_SIM   = DATA_DIR / "state_sim.json"
STATE_FILE_LIVE  = DATA_DIR / "state_live.json"
RESULTS_FILE     = DATA_DIR / os.getenv("RESULTS_FILE", "backtest_results.json")
ML_DATASET_FILE  = DATA_DIR / "ml_dataset.csv"
AI_MODEL_FILE    = DATA_DIR / "ai_model.joblib"
AI_MODEL_FILE    = DATA_DIR / "ai_model.joblib"
NEURAL_MODEL_FILE = DATA_DIR / "neural_model.joblib"
CHECKPOINT_DB    = DATA_DIR / "checkpoint.db"
MOCK_ANCHOR_FILE = DATA_DIR / "mock_anchor.json"
TRADE_JOURNAL_FILE = DATA_DIR / "trade_journal.csv"
DATA_CACHE_DIR   = DATA_DIR / "cache"
TRAINING_LOG_FILE = DATA_DIR / "training_history.csv"
MODEL_SNAPSHOTS_DIR = DATA_DIR / "model_snapshots"   # ❄️ Versiones guardadas de la IA

# ─── Timing del bot ─────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC  = 1    # Evaluar cada 1 segundo en LIVE (entre velas de 5min)
REST_INTERVAL_SEC  = 1    # Descanso fuera de ventana de trading (1s para no bloquear en LIVE)
SESSION_PAUSE      = 0    # Segundos entre sesiones de simulación (0 = máxima velocidad)


# ─── Parámetros de Estrategia (Refactorización) ──────────────────────────────
MIN_BARS_REQUIRED   = 80
EMA_MEDIUM_PERIOD   = 20
EMA_LONG_PERIOD     = 200
ADX_PERIOD          = 14
VOLUME_ROLLING_WIN  = 20
REGIME_LOG_INTERVAL = 10  # segundos

# Umbrales de Señal
VELEZ_BOUNCE_MULT    = 1.002
VWAP_BOUNCE_ZSCORE   = -2.0
EMERGENCY_ZSCORE     = -2.5
TARGET_MIN_NET_PROFIT_USD = 3.0  # Mínimo $3.00 libres por trade para vencer comisiones (Modo Quirúrgico)
EMA_CROSS_RSI_MIN    = 35
EMA_CROSS_RSI_MAX    = 70

# Filtros de Calidad
QUALITY_RSI_MIN      = 43
QUALITY_RSI_MAX      = 47
QUALITY_ADX_THRESHOLD = 22  # Punto Dulce: Suficiente fuerza para ganar, suficiente filtro para proteger.
QUALITY_CHOP_THRESHOLD = 60 # Bloqueo total si Choppiness > 60
QUALITY_ZSCORE_MIN   = -2.3
CHOP_PERIOD          = 14
CONFIDENCE_THRESHOLD = 0.55
MIN_SAMPLES          = 10

# ─── Heurísticas de IA (Cold-Start) ──────────────────────────────────────────
# Parámetros para neural_filter.py cuando hay < 10 muestras
COLD_START_MIN_SAMPLES    = 10

COLD_START_BASE_SCORE     = 0.58

# Umbrales Heurísticos
HEURISTIC_RSI_UPPER       = 72
HEURISTIC_MACD_MIN        = 0.1
HEURISTIC_CONF_HIGH       = 5
HEURISTIC_CONF_MID        = 4
HEURISTIC_VOL_MIN         = 0.7
HEURISTIC_ATR_MAX         = 0.7

# Ajustes de Score (Bloques de probabilidad)
HEURISTIC_PENALTY_RSI     = -0.25
HEURISTIC_PENALTY_MACD    = -0.15
HEURISTIC_PENALTY_ATR     = -0.15
HEURISTIC_BOOST_CONF_H    = 0.25
HEURISTIC_BOOST_CONF_M    = 0.15
HEURISTIC_BOOST_VOL       = 0.15
HEURISTIC_BOOST_TREND     = 0.15


# ─── Settlement T+1 ─────────────────────────────────────────────────────────
SETTLEMENT_DAYS = 1

# ─── Orquestador y Auto-Trainer ──────────────────────────────────────────────
# Límite máximo de bots que el Auto-Trainer puede lanzar en paralelo (Safety Cap: 10)
MAX_AUTOTRAINER_BOTS = min(10, int(os.getenv("MAX_AUTOTRAINER_BOTS", "4")))
# Buffer de seguridad de RAM que siempre debe quedar libre en el sistema (MB)
# Reducido a 512MB para ser realista con el uso de memoria en Mac
MIN_FREE_RAM_MB = int(os.getenv("MIN_FREE_RAM_MB", "512"))
