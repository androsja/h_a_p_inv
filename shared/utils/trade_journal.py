"""
utils/trade_journal.py ─ Bitácora completa de trades para análisis y mejora de algoritmos.

Guarda CADA trade cerrado con toda la información disponible:
  - Contexto del mercado al momento de la entrada (régimen, indicadores)
  - Parámetros de gestión de riesgo (SL, TP, ATR)
  - Resultado real (PnL bruto, neto, porcentaje, ganador/perdedor)
  - Cuántas barras duró el trade
  - Motivo de salida
  - Estrategia que generó la señal

Esto permite responder preguntas como:
  ¿En qué regímenes gana más la estrategia A?
  ¿Con qué ADX promedio los trades de MSFT son rentables?
  ¿El R/R real es el esperado o se aleja?

Archivo: data/trade_journal.csv
"""

import csv
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from shared import config

JOURNAL_PATH = config.TRADE_JOURNAL_FILE
_journal_lock = threading.Lock()

# Columnas completas — en este orden
COLUMNS = [
    # ── Identificación ──────────────────────────────────────────
    "timestamp_close",   # Fecha/hora de cierre del trade (UTC)
    "symbol",            # Ticker (AAPL, MSFT, etc.)
    "session",           # Número de sesión del simulador

    # ── Estrategia ───────────────────────────────────────────────
    "strategy",          # A = Oliver Vélez, B = VWAP bounce, MOMENTUM = breakout
    "regime",            # Régimen al momento de entrada: TREND_UP, RANGE, etc.

    # ── Parámetros de entrada ────────────────────────────────────
    "entry_price",       # Precio de compra
    "exit_price",        # Precio de venta
    "qty",               # Cantidad de acciones
    "stop_loss",         # Stop Loss configurado
    "take_profit",       # Take Profit configurado
    "hold_bars",         # Cuántas iteraciones duró el trade abierto

    # ── Resultado ────────────────────────────────────────────────
    "exit_reason",       # TAKE_PROFIT, STOP_LOSS, TRAILING_STOP, FORCED_SELL
    "gross_pnl",         # PnL bruto (sin fees/slippage)
    "gross_pnl_pct",     # PnL bruto % sobre capital invertido
    "is_win",            # 1 = ganador, 0 = perdedor

    # ── Ratio R/R real ───────────────────────────────────────────
    "risk_distance",     # Diferencia entrada - stop_loss (riesgo asumido)
    "reward_distance",   # Diferencia take_profit - entrada (recompensa esperada)
    "rr_ratio",          # reward_distance / risk_distance (ratio R/R esperado)
    "achieved_rr",       # (exit_price - entry_price) / risk_distance (R/R real)

    # ── Indicadores al momento de entrada ────────────────────────
    "rsi",               # RSI al entrar
    "macd_hist",         # Histograma MACD al entrar
    "adx",               # ADX al entrar (fuerza de tendencia)
    "atr_pct",           # ATR como % del precio
    "vwap_dist_pct",     # % de distancia al VWAP
    "ema_diff_pct",      # % diferencia EMA rápida - lenta
    "vol_ratio",         # Volumen / promedio 20 barras
    "zscore_vwap",       # Z-Score VWAP (dist estadística del precio al VWAP)
]


def _ensure_header():
    """Crea el archivo con cabecera si no existe aún."""
    with _journal_lock:
        if not JOURNAL_PATH.exists():
            JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()


def record_trade(
    symbol: str,
    session: int,
    entry_price: float,
    exit_price: float,
    qty: float,
    stop_loss: float,
    take_profit: float,
    hold_bars: int,
    exit_reason: str,
    ml_features: dict,
    timestamp: str | None = None,
):
    """
    Registra un trade cerrado en el diario completo.

    Args:
        symbol       -- Ticker del activo
        session      -- Número de sesión actual
        entry_price  -- Precio al que se compró
        exit_price   -- Precio al que se vendió
        qty          -- Cantidad operada
        stop_loss    -- Stop loss configurado
        take_profit  -- Take profit configurado
        hold_bars    -- Iteraciones que estuvo abierto el trade
        exit_reason  -- Motivo de salida (TAKE_PROFIT, STOP_LOSS, etc.)
        ml_features  -- Dict con indicadores y regime del momento de entrada
    """
    try:
        _ensure_header()

        gross_pnl = (exit_price - entry_price) * qty
        invested = entry_price * qty
        gross_pnl_pct = (gross_pnl / invested * 100) if invested > 0 else 0
        is_win = 1 if gross_pnl > 0 else 0

        risk_distance   = entry_price - stop_loss
        reward_distance = take_profit - entry_price
        rr_ratio    = (reward_distance / risk_distance) if risk_distance > 0 else 0
        achieved_rr = ((exit_price - entry_price) / risk_distance) if risk_distance > 0 else 0

        # Detectar estrategia desde las confirmaciones guardadas en ml_features (si existen)
        strategy = "?"
        regime = ml_features.get("regime", "NEUTRAL")
        confirmations_raw = ml_features.get("confirmations", [])
        if isinstance(confirmations_raw, str):
            try:
                confirmations_raw = json.loads(confirmations_raw)
            except Exception:
                confirmations_raw = []
        for c in confirmations_raw:
            if "Vélez" in str(c) or "Estrategia A" in str(c):
                strategy = "A"
                break
            elif "VWAP" in str(c) or "Estrategia B" in str(c):
                strategy = "B"
                break
            elif "MOMENTUM" in str(c):
                strategy = "MOMENTUM"
                break

        row = {
            "timestamp_close": timestamp if timestamp else datetime.now(timezone.utc).isoformat(),
            "symbol":          symbol,
            "session":         session,
            "strategy":        strategy,
            "regime":          regime,
            "entry_price":     round(entry_price, 4),
            "exit_price":      round(exit_price, 4),
            "qty":             round(qty, 6),
            "stop_loss":       round(stop_loss, 4),
            "take_profit":     round(take_profit, 4),
            "hold_bars":       hold_bars,
            "exit_reason":     exit_reason,
            "gross_pnl":       round(gross_pnl, 4),
            "gross_pnl_pct":   round(gross_pnl_pct, 4),
            "is_win":          is_win,
            "risk_distance":   round(risk_distance, 4),
            "reward_distance": round(reward_distance, 4),
            "rr_ratio":        round(rr_ratio, 3),
            "achieved_rr":     round(achieved_rr, 3),
            "rsi":             round(ml_features.get("rsi", 0), 2),
            "macd_hist":       round(ml_features.get("macd_hist", 0), 6),
            "adx":             round(ml_features.get("adx", 0), 2),
            "atr_pct":         round(ml_features.get("atr_pct", 0), 4),
            "vwap_dist_pct":   round(ml_features.get("vwap_dist_pct", 0), 4),
            "ema_diff_pct":    round(ml_features.get("ema_diff_pct", 0), 4),
            "vol_ratio":       round(ml_features.get("vol_ratio", 1.0), 3),
            "zscore_vwap":     round(ml_features.get("zscore_vwap", 0), 3),
        }

        with _journal_lock:
            with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writerow(row)
        
        # 🔗 Notificar al state_writer para actualización en tiempo real del Dashboard
        try:
            from shared.utils import state_writer
            state_writer.record_trade(
                symbol=symbol,
                side="BUY/SELL", # Genérico para el historial rápido
                price=exit_price,
                qty=qty,
                pnl=gross_pnl,
                timestamp=timestamp
            )
        except Exception:
            pass

    except Exception as e:
        from shared.utils.logger import log
        log.error(f"[trade_journal] Error guardando trade: {e}")
