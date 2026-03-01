"""
config.py ─ Configuración central del sistema de trading
Carga variables del .env y provee constantes globales.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Cargar .env ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
# Si este archivo está dentro de 'shared/', subir un nivel para encontrar el root
if BASE_DIR.name == "shared":
    BASE_DIR = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")

# URL base de la API de Hapi Trade
HAPI_BASE_URL   = "https://api.hapitrade.com/v1"

# ─── Credenciales Alpaca (para datos de mercado) ──────────────────────────
ALPACA_API_KEY      = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY   = os.getenv("APCA_API_SECRET_KEY", "")
ALPACA_BASE_URL     = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

# ─── Modo de operación ──────────────────────────────────────────────────────
TRADING_MODE    = os.getenv("TRADING_MODE", "SIMULATED").upper()   # LIVE | SIMULATED
HAPI_TEST_MODE  = os.getenv("HAPI_IS_TEST_MODE", "true").lower() == "true"

# ─── Gestión de riesgo (Intraday) ───────────────────────────────────────────
STOP_LOSS_PCT       = float(os.getenv("STOP_LOSS_PCT",    "0.025"))   # 2.5% SL
TAKE_PROFIT_PCT     = float(os.getenv("TAKE_PROFIT_PCT",  "0.05"))    # 5.0% TP  → ratio 1:2
MAX_POSITION_USD    = float(os.getenv("MAX_POSITION_USD", "10000"))   # Relajado a tamaño total de la cuenta
MAX_RISK_PCT        = float(os.getenv("MAX_RISK_PCT", "0.025"))      # Máximo riesgo de 2.5% del capital por trade
CLEARING_COST_USD   = float(os.getenv("CLEARING_COST_USD","0.15"))   # $0.15 Apex

# ─── Latencia Bogotá → Nueva York ───────────────────────────────────────────
LATENCY_OFFSET_CENTS = float(os.getenv("LATENCY_OFFSET_CENTS", "2")) / 100

# ─── Logs ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE  = BASE_DIR / os.getenv("LOG_FILE", "logs/trading_bot.log")

# ─── Estrategia Intraday (velas de 5 minutos) ────────────────────────────────
# EMA 12/26 es el estándar para intraday (mismos períodos que el MACD clásico)
EMA_FAST       = 12     # EMA rápida  — señal de entrada
# Hyper-Aggressive V3: Ajustado a 0.55 para mayor cautela (Anti-Trampas).
CONFIDENCE_THRESHOLD = 0.55
EMA_SLOW       = 26     # EMA lenta   — tendencia de fondo
RSI_PERIOD     = 14     # RSI estándar
RSI_OVERBOUGHT = 75     # Más agresivo: permite tendencias fuertes (antes 70)
RSI_OVERSOLD   = 40     # Más agresivo: entra más temprano en rebotes (antes 35)

# ─── Datos de mercado ────────────────────────────────────────────────────────
# Intervalo y período para yfinance (intraday en lugar de scalping)
DATA_INTERVAL  = "5m"   # Regresando a 5m para mayor estabilidad, manteniendo 20 activos.
DATA_PERIOD    = "60d"  # 60 días de historial → ~7,800 velas por símbolo

# ─── Ventana de trading: apertura de NY (mayor liquidez intraday) ────────────
# 🕒 Ventana de Trading (Horario NY)
# Hiper-Agresivo V3: Expandimos la ventana de 2 a 4 horas.
TRADING_OPEN_HOUR   = 9    # Hora de apertura (NY)
TRADING_OPEN_MIN    = 30
TRADING_CLOSE_HOUR  = 13   # Hora de cierre de ventana
TRADING_CLOSE_MIN   = 30

# ─── Archivos de datos ──────────────────────────────────────────────────────
ASSETS_FILE      = BASE_DIR / "assets.json" # Legacy
ASSETS_FILE_SIM  = BASE_DIR / "assets_sim.json"
ASSETS_FILE_LIVE = BASE_DIR / "assets_live.json"
DATA_DIR         = BASE_DIR / "data"
COMMAND_FILE     = DATA_DIR / "command.json"
STATE_FILE       = DATA_DIR / "state.json" # Legacy/Default
STATE_FILE_SIM   = DATA_DIR / "state_sim.json"
STATE_FILE_LIVE  = DATA_DIR / "state_live.json"
RESULTS_FILE     = DATA_DIR / "backtest_results.json"
ML_DATASET_FILE  = DATA_DIR / "ml_dataset.csv"
AI_MODEL_FILE    = DATA_DIR / "ai_model.joblib"
AI_MODEL_FILE    = DATA_DIR / "ai_model.joblib"
NEURAL_MODEL_FILE = DATA_DIR / "neural_model.joblib"
CHECKPOINT_DB    = DATA_DIR / "checkpoint.db"
MOCK_ANCHOR_FILE = DATA_DIR / "mock_anchor.json"
TRADE_JOURNAL_FILE = DATA_DIR / "trade_journal.csv"
DATA_CACHE_DIR   = DATA_DIR / "cache"
MODEL_SNAPSHOTS_DIR = DATA_DIR / "model_snapshots"   # ❄️ Versiones guardadas de la IA

# ─── Timing del bot ─────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC  = 1    # Evaluar cada 1 segundo en LIVE (entre velas de 5min)
REST_INTERVAL_SEC  = 60   # Descanso fuera de ventana de trading

# ─── Settlement T+1 ─────────────────────────────────────────────────────────
SETTLEMENT_DAYS = 1
