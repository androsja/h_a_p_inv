import argparse
import copy
import json
import logging
import os
import threading
import time

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
    threads = []
    stop_event = threading.Event()
    
    for sym in force_symbols:
        # Cada hilo necesita su propia instancia de argumentos y broker
        thread_args = copy.copy(args)
        thread_args.symbol = sym
        thread_broker = init_broker_fn(thread_args)
        
        t = threading.Thread(
            target=run_bot_fn, 
            args=(thread_broker, thread_args, session_num, stop_event),
            name=f"Worker-{sym}"
        )
        t.daemon = True
        t.start()
        threads.append(t)
    
    # El hilo principal espera y vigila command.json para abortar o resetear
    try:
        while True:
            time.sleep(2)
            if os.path.exists("/app/data/command.json"):
                with open("/app/data/command.json") as f:
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
