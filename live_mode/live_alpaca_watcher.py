"""
live_mode/live_alpaca_watcher.py ─ Watcher para Alpaca Paper Trading.

Proceso independiente que monitorea command.json y lanza/detiene hilos
de run_live_bot (Alpaca Paper) cuando recibe live_start / live_stop.

NO toca simulation_mode. NO modifica main_sim.py.
Solo usa shared/ como insumos de lectura (estrategia, ML, risk_manager).
"""

import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_alpaca_watcher")

def main():
    log.info("🦙 Live Alpaca Watcher iniciado. Esperando comandos live_start/live_stop...")
    try:
        from live_mode.alpaca_launcher import watch_commands
        watch_commands()
    except KeyboardInterrupt:
        log.info("Watcher detenido por el usuario.")
    except Exception as e:
        log.error(f"Error en watcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
