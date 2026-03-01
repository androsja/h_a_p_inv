"""
utils/logger.py ─ Sistema de logging estructurado para el bot de trading.

Características:
  • Salida en consola con colores (INFO=cyan, WARNING=amarillo, ERROR=rojo)
  • Rotación de archivo de log (máx. 5 MB, 3 archivos de respaldo)
  • Función específica para registrar intentos de orden y errores 429
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from colorama import Fore, Style, init as colorama_init
from shared import config

colorama_init(autoreset=True)


# ─── Formateador con colores para la consola ────────────────────────────────
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    Fore.WHITE,
        logging.INFO:     Fore.CYAN,
        logging.WARNING:  Fore.YELLOW,
        logging.ERROR:    Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        base = super().format(record)
        return f"{color}{base}{Style.RESET_ALL}"


# ─── Configuración global ───────────────────────────────────────────────────
def setup_logger(name: str = "trading_bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # Evitar duplicar handlers si ya fue inicializado
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ── Handler de consola ──────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(fmt, datefmt=date_fmt))
    logger.addHandler(console_handler)

    # ── Handler de archivo con rotación ────────────────────────────────────
    log_path: Path = config.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(file_handler)

    return logger


def set_symbol_log(symbol: str) -> None:
    """Cambia el archivo de log para que guarde un historial dedicado por símbolo."""
    global log
    # Remover viejos FileHandlers individuales de símbolos
    handlers_to_remove = []
    for h in log.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            # No tocar el log principal (trading_bot.log)
            if "historial_" in str(h.baseFilename):
                handlers_to_remove.append(h)
                
    for h in handlers_to_remove:
        log.removeHandler(h)
    
    # Crear nuevo handler exclusivo para este símbolo
    sym_log_path = Path(f"logs/historial_{symbol}.log")
    sym_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=sym_log_path,
        maxBytes=10 * 1024 * 1024,   # 10 MB per symbol should be plenty
        backupCount=1,
        encoding="utf-8",
    )
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    
    log.addHandler(file_handler)
    log.info(f"📁 --- INICIANDO BITÁCORA AISLADA PARA {symbol} ---")


# ─── Logger principal del bot ───────────────────────────────────────────────
log = setup_logger("trading_bot")


# ─── Helpers para eventos de trading ────────────────────────────────────────
def log_order_attempt(symbol: str, side: str, price: float, qty: float, reason: str) -> None:
    """Registra un intento de colocación de orden."""
    log.info(
        f"ORDER_ATTEMPT | {side.upper():4s} {symbol} | "
        f"qty={qty:.4f} | price=${price:.2f} | reason={reason}"
    )


def log_order_filled(symbol: str, side: str, fill_price: float, qty: float, pnl: float | None = None, timestamp: str | None = None, reason: str = "", metadata: dict = None) -> None:
    """Registra la confirmación de ejecución de una orden."""
    pnl_str = f"| PnL=${pnl:+.2f}" if pnl is not None else ""
    log.info(
        f"ORDER_FILLED  | {side.upper():4s} {symbol} | "
        f"qty={qty:.4f} | fill_price=${fill_price:.2f} {pnl_str}"
    )
    try:
        from shared.utils.state_writer import record_trade
        record_trade(
            symbol, side, fill_price, qty, pnl if pnl is not None else 0.0, 
            timestamp=timestamp, reason=reason, metadata=metadata
        )
    except Exception as e:
        log.error(f"Error registrando trade en state.json: {e}")


def log_rate_limit(endpoint: str, retry_after: int | None = None) -> None:
    """Registra un error 429 (Rate Limit) de la API de Hapi."""
    retry_str = f"Retry-After={retry_after}s" if retry_after else "sin cabecera Retry-After"
    log.warning(
        f"RATE_LIMIT_429 | Endpoint={endpoint} | {retry_str} | "
        f"El bot hará pausa antes de reintentar."
    )


def log_risk_block(symbol: str, reason: str) -> None:
    """Registra que la gestión de riesgo bloqueó una operación."""
    log.warning(f"RISK_BLOCK | {symbol} | Razón: {reason}")


def log_settlement_block(symbol: str, needed: float, available: float) -> None:
    """Registra que el capital está bloqueado por settlement T+1."""
    log.warning(
        f"SETTLEMENT_BLOCK | {symbol} | "
        f"Necesario=${needed:.2f} | Disponible=${available:.2f} | "
        f"El resto está en liquidación T+1 y no se puede usar hoy."
    )


def log_market_closed(next_open: str) -> None:
    """Registra que el mercado NYSE está cerrado."""
    log.info(f"MARKET_CLOSED | Próxima apertura estimada: {next_open}")
