"""
main_sim.py ─ Motor principal del Bot de Trading Automatizado (Modo Simulación).
"""

import sys
import argparse
import os
import json
from shared import config
from shared.utils.logger import log
from shared.broker.interface import BrokerInterface
from shared.broker.hapi_mock import HapiMock
from shared.engine.runner import SimulationRunner
from shared.engine.trading_engine import TradingEngine

BANNER = """
╔══════════════════════════════════════════════════════╗
║     🇨🇴 HAPI SCALPING BOT  ─  Colombia Edition       ║
║     Estrategia: Multi-Strategy Orchestrator          ║
║     Módulo de Simulación Refactorizado v2.0          ║
╚══════════════════════════════════════════════════════╝
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot de Trading - Modo SIMULACIÓN")
    parser.add_argument("--symbol", default=None, help="Símbolo específico para simular.")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Capital inicial ($)")
    p = parser.parse_args()
    p.mode = "SIMULATED"
    
    fixed = os.getenv("FIXED_SYMBOL", "").strip()
    if fixed and not p.symbol:
        p.symbol = fixed
    return p

def init_broker_callback(args: argparse.Namespace, **kwargs) -> BrokerInterface:
    """Callback para inicializar el bróker (Mock) basado en comandos o args."""
    is_live_paper = False
    sim_start_date = None
    
    try:
        cmd_file = config.COMMAND_FILE
        if cmd_file.exists():
            with open(cmd_file) as f:
                cmds = json.load(f)
            is_live_paper = cmds.get("force_paper_trading", False)
            sim_start_date = cmds.get("sim_start_date")
            
            # Ajuste de símbolo forzado si aplica
            force_symbol = cmds.get("force_symbol", "")
            if is_live_paper and force_symbol and force_symbol != "AUTO":
                args.symbol = force_symbol
    except: pass

    return HapiMock(
        symbol=args.symbol, 
        initial_cash=args.cash, 
        live_paper=is_live_paper, 
        start_date=sim_start_date
    )

def run_bot_compat(broker, args, session_num=1, stop_event=None, asset_type="normal"):
    """Wrapper para compatibilidad con hilos paralelos de Live Paper."""
    engine = TradingEngine(broker, args)
    engine.run(session_num=session_num, stop_event=stop_event, asset_type=asset_type)

def main():
    print(BANNER)
    args = parse_args()
    
    # Configuración de archivos de estado
    from shared.utils.state_writer import set_state_file
    from shared.data.market_data import set_assets_file
    set_state_file(config.STATE_FILE_SIM)
    set_assets_file(config.ASSETS_FILE_SIM)
    
    # El Runner se encarga de todo el ciclo de vida
    runner = SimulationRunner(init_broker_callback, run_bot_compat)
    
    try:
        runner.main_loop(args)
    except KeyboardInterrupt:
        log.info("\n🛑 Proceso detenido por el usuario (Ctrl+C).")
    except Exception as e:
        log.critical(f"💥 Error fatal en el sistema: {e}", exc_info=True)
    finally:
        log.info("👋 Simulación finalizada.")

if __name__ == "__main__":
    main()
