"""
live_mode/runner_live.py ─ Orquestador Exclusivo de Live y Live Paper Trading.
Diseñado para correr estrictamente de manera concurrente (Parallel Bots).
"""

import os
import json
import importlib
import argparse
import time
from shared import config
from shared.utils.logger import log
from shared.data import market_data
from shared.utils.state_writer import update_state, clear_state
from shared.utils.live_paper_launcher import launch_parallel_bots
from shared.engine.utils import SessionInterrupted, smart_sleep

class LiveRunner:
    """
    Gestiona la ejecución paralela e indefinida de bots de trading en tiempo real.
    Ninguna lógica secuencial o de simulación histórica se encuentra aquí.
    """
    def __init__(self, init_broker_fn, run_bot_fn):
        self.init_broker = init_broker_fn
        self.run_bot_logic = run_bot_fn
        self.session_num = 1 # Live no itera por sesiones de la misma manera
        
    def main_loop(self, args: argparse.Namespace):
        log.info("🌐 MOTOR LIVE/PAPER TRADING: Preparando conexión asíncrona...")
        
        while True:
            try:
                # 1. Comandos Globales y Recarga de Símbolos
                if self._process_global_commands():
                    continue 

                # 2. Re-cargar símbolos activos del Gestor (fuente de verdad)
                self._reload_symbols()
                
                from shared.data.market_data import get_symbols
                active_symbols = get_symbols()
                
                if not active_symbols:
                    log.warning("⚠️ No hay símbolos activos en el gestor. Esperando...")
                    smart_sleep(10)
                    continue

                log.info(f"🚀 Lanzando Bots Paralelos para {len(active_symbols)} símbolos...")
                
                # Despachamos todos los bots en paralelo
                # launch_parallel_bots encapsula la espera infinita en un pool de hilos
                launch_parallel_bots(
                    args, 
                    active_symbols, 
                    self.session_num, 
                    self.run_bot_logic, 
                    self.init_broker, 
                    save_checkpoint_fn=None # Live no usa checkpoints históricos
                )

                # Si logramos salir de aquí sin error, significa que hubo una interrupción voluntaria
                break

            except SessionInterrupted:
                log.info("📢 Interrupción Live detectada (Reinicio/Cambio). Reiniciando...")
                smart_sleep(2)
                continue
            except Exception as e:
                log.error(f"❌ Error crítico en bucle LIVE: {e}", exc_info=True)
                smart_sleep(5)

    def _reload_symbols(self):
        try:
            importlib.reload(market_data)
            # Para Live, el market data setter ya debió inicializarse en main_live.py,
            # pero nos aseguramos invocando la recarga para que lea assets_live.json
        except Exception as e:
            log.warning(f"⚠️ Error recargando mercado para Live: {e}")

    def _process_global_commands(self):
        cmd_file = config.COMMAND_FILE
        if not cmd_file.exists(): return False
        
        try:
            with open(cmd_file, "r") as f: cmds = json.load(f)
            
            # Freeze/Unfreeze
            frozen = cmds.get("strategy_frozen", False)
            try:
                from shared.utils.neural_filter import _filter_registry
                for nf in list(_filter_registry.values()):
                    if frozen and not nf.is_frozen: nf.freeze()
                    elif not frozen and nf.is_frozen: nf.unfreeze()
            except: pass

            if cmds.get("reset_all") or cmds.get("restart_sim"):
                log.info(f"🔄 Reiniciando entorno LIVE...")
                cmds["reset_all"] = False
                cmds["restart_sim"] = False
                with open(cmd_file, "w") as f: json.dump(cmds, f)
                
                clear_state()
                update_state("─", mode="LIVE", status="restarting")
                return True
        except: pass
        return False
