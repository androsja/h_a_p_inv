import argparse
import copy
import json
import logging
import os
import threading
import time

import config
log = logging.getLogger("trading_bot")

def launch_parallel_bots(args: argparse.Namespace, force_symbols: list[str], session_num: int, run_bot_fn, init_broker_fn, checkpoint_fn=None) -> None:
    """
    Lanza el bot en modalidad multihilo para Live Paper (un hilo por símbolo).
    """
    # ── Guardar el progreso de la simulación ANTES de cambiar a Live Paper ──
    if checkpoint_fn and getattr(args, 'mode', 'SIMULATED') == 'SIMULATED':
        try:
            # We don't have the exact symbol_idx here, but we can pass 0 or a placeholder
            checkpoint_fn(0, args.symbol or "", session_num)
            log.info(f"💾 CHECKPOINT guardado antes de iniciar Live Paper paralelo.")
        except Exception as _e:
            log.warning(f"Error guardando checkpoint: {_e}")

    log.info(f"🚀 Iniciando monitoreo PARALELO para {len(force_symbols)} símbolo(s)...")
    
    # ── Limpiar el flag reset_all ANTES de lanzar los hilos ─────────────────
    # El dashboard escribe reset_all:true al iniciar Live Paper.
    # Si los hilos lo leen durante su pausa de 1s, se autodestruyen.
    cmd_file = config.COMMAND_FILE
    try:
        if os.path.exists(cmd_file):
            with open(cmd_file) as f:
                cmd_data = json.load(f)
            if cmd_data.get("reset_all"):
                cmd_data["reset_all"] = False
                with open(cmd_file, "w") as f:
                    json.dump(cmd_data, f)
                log.info("🧹 Flag reset_all limpiado antes de iniciar hilos.")
    except Exception as _ce:
        log.warning(f"No se pudo limpiar reset_all: {_ce}")

    threads = []
    stop_event = threading.Event()
    
    for sym in force_symbols:
        # Cada hilo necesita su propia instancia de argumentos y broker
        thread_args = copy.copy(args)
        thread_args.symbol = sym
        thread_broker = init_broker_fn(thread_args, is_live_paper_override=True)
        
        t = threading.Thread(
            target=run_bot_fn, 
            args=(thread_broker, thread_args, session_num, stop_event),
            name=f"Worker-{sym}"
        )
        t.daemon = True
        t.start()
        threads.append(t)
    
    # El hilo principal espera y vigila command.json para abortar o resetear.
    # Esperamos 5s antes del primer chequeo para evitar capturar el reset_all
    # de arranque (que ya fue limpiado anterior pero puede haber race condition).
    try:
        time.sleep(5)  # Gracia inicial — ignora señales del momento de arranque
        while True:
            time.sleep(2)
            if os.path.exists(cmd_file):
                with open(cmd_file) as f:
                    c = json.load(f)
                if c.get("reset_all") or c.get("force_paper_trading") is False:
                    log.info("🛑 Deteniendo todos los hilos paralelos...")
                    stop_event.set()
                    break
    except KeyboardInterrupt:
        stop_event.set()
    
    log.info("⏳ Esperando a que todos los hilos terminen...")
    for t in threads:
        t.join(timeout=5)
    log.info("✅ Todos los hilos cerrados exitosamente.")
