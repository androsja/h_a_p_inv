"""
live_mode/alpaca_launcher.py ─ Lanzador de hilos Alpaca Paper Trading.

Arranca N hilos en paralelo (uno por símbolo activo en assets_live.json).
Responde a comandos del dashboard para iniciar/detener.
NO toca simulation_mode en absoluto.
"""

import json
import threading
import time
from pathlib import Path

from shared import config
from shared.utils.logger import log


# ── Estado global de los hilos ────────────────────────────────────────────────
_threads: dict[str, threading.Thread] = {}     # {symbol: Thread}
_stop_events: dict[str, threading.Event] = {}  # {symbol: Event}
_global_stop = threading.Event()


def get_active_symbols() -> list[str]:
    """Lee los símbolos habilitados de assets.json (archivo maestro compartido)."""
    try:
        path = config.ASSETS_FILE  # ← usa el mismo archivo que el Gestor de Símbolos
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Formato estándar: {"assets": [{"symbol": "X", "enabled": true}, ...]}
            assets = data.get("assets", [])
            if assets and isinstance(assets[0], dict):
                return [a["symbol"] for a in assets if a.get("enabled", False)]
    except Exception as e:
        log.warning(f"[AlpacaLauncher] Error leyendo assets.json: {e}")
    return []


from shared import config

def launch(api_key: str, secret_key: str, initial_cash: float = config.INITIAL_CASH_LIVE, live_mode_type: str = "PAPER"):
    """
    Lanza hilos de trading en vivo para todos los símbolos activos.
    Si ya hay hilos corriendo, los detiene primero.
    """
    from live_mode.run_live_bot import run_symbol_live
    from shared.utils.state_writer import set_state_file, clear_state

    # Asegurar que Live escribe en state_live.json (NUNCA state_sim.json)
    set_state_file(config.STATE_FILE_LIVE)

    symbols = get_active_symbols()
    if not symbols:
        log.warning("[AlpacaLauncher] No hay símbolos activos en assets.json. Nada que lanzar.")
        return

    # ── LIMPIAR estado anterior para que el dashboard solo muestre los nuevos símbolos ──
    # Sin esto, los símbolos de sesiones anteriores siguen apareciendo en la tabla.
    log.info("[AlpacaLauncher] 🧹 Limpiando state_live.json de sesión anterior...")
    clear_state()

    # Detener hilos previos si los hay
    stop_all()

    log.info(f"[AlpacaLauncher] 🚀 Lanzando {len(symbols)} hilo(s): {symbols}")

    for i, symbol in enumerate(symbols):
        stop_ev = threading.Event()
        _stop_events[symbol] = stop_ev

        t = threading.Thread(
            target=run_symbol_live,
            kwargs={
                "symbol":       symbol,
                "api_key":      api_key,
                "secret_key":   secret_key,
                "initial_cash": initial_cash,
                "stop_event":   stop_ev,
                "session_num":  i + 1,
                "live_mode_type": live_mode_type,
            },
            name=f"live_{symbol}",
            daemon=True,
        )
        _threads[symbol] = t
        t.start()
        log.info(f"[AlpacaLauncher] ✅ Hilo [{symbol}] iniciado")


def stop_all():
    """Envía stop_event a todos los hilos activos y espera que terminen."""
    for symbol, ev in list(_stop_events.items()):
        ev.set()
        log.info(f"[AlpacaLauncher] 🛑 Señal de parada enviada a [{symbol}]")

    for symbol, t in list(_threads.items()):
        t.join(timeout=5)
        log.info(f"[AlpacaLauncher] [{symbol}] hilo terminado")

    _threads.clear()
    _stop_events.clear()


def is_running() -> bool:
    """Retorna True si hay algún hilo de live corriendo."""
    return any(t.is_alive() for t in _threads.values())


def status() -> dict:
    """Retorna el estado de todos los hilos activos."""
    return {
        sym: t.is_alive()
        for sym, t in _threads.items()
    }


def watch_commands():
    """
    Loop que monitorea command.json buscando comandos de live.
    Corre en su propio hilo desde el servidor o desde main_live.py.
    """
    log.info("[AlpacaLauncher] 👁️ Monitor de comandos iniciado")
    api_key    = config.ALPACA_API_KEY
    secret_key = config.ALPACA_SECRET_KEY

    while not _global_stop.is_set():
        try:
            if config.COMMAND_FILE.exists():
                with open(config.COMMAND_FILE) as f:
                    cmds = json.load(f)

                if cmds.get("live_start"):
                    log.info("[AlpacaLauncher] 📥 Comando live_start recibido")
                    mode = cmds.get("live_mode", "PAPER")
                    launch(api_key, secret_key, live_mode_type=mode)
                    # Limpiar flag
                    cmds["live_start"] = False
                    with open(config.COMMAND_FILE, "w") as f:
                        json.dump(cmds, f)

                elif cmds.get("live_stop"):
                    log.info("[AlpacaLauncher] 📥 Comando live_stop recibido")
                    stop_all()
                    cmds["live_stop"] = False
                    with open(config.COMMAND_FILE, "w") as f:
                        json.dump(cmds, f)

                elif cmds.get("reset_neural"):
                    log.info("[AlpacaLauncher] 🧨 Comando reset_neural recibido — limpiando IA en memoria...")
                    # Detener hilos para que no sigan escribiendo el modelo
                    stop_all()
                    # Resetear el singleton de la Red Neuronal en memoria
                    try:
                        from shared.utils.neural_filter import reset_neural_filter
                        reset_neural_filter()
                        log.info("[AlpacaLauncher] ✅ Red Neuronal reseteada en memoria y disco.")
                    except Exception as e:
                        log.error(f"[AlpacaLauncher] Error reseteando neural filter: {e}")
                    # Limpiar flag
                    cmds["reset_neural"] = False
                    with open(config.COMMAND_FILE, "w") as f:
                        json.dump(cmds, f)

        except Exception as e:
            log.debug(f"[AlpacaLauncher] watch_commands error: {e}")

        time.sleep(2)

