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
from simulation_mode.runner_sim import SimulationRunner
from shared.engine.trading_engine import TradingEngine
from shared.engine.utils import SessionInterrupted

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
    is_live_paper = kwargs.get('is_live_paper_override', False)
    sim_start_date = None
    sim_end_date = None
    
    # Override date from modern Docker environment vars primarily
    env_start = os.getenv("HAPI_SIM_START_DATE")
    if env_start and env_start.strip():
        sim_start_date = env_start.strip()
        
    env_end = os.getenv("HAPI_SIM_END_DATE")
    if env_end and env_end.strip():
        sim_end_date = env_end.strip()

    if not sim_start_date or not sim_end_date:
        try:
            cmd_file = config.COMMAND_FILE
            if cmd_file.exists():
                with open(cmd_file) as f:
                    cmds = json.load(f)
                
                if not is_live_paper:
                    is_live_paper = cmds.get("force_paper_trading", False)
                
                if not sim_start_date:
                    sim_start_date = cmds.get("sim_start_date")
                if not sim_end_date:
                    sim_end_date = cmds.get("sim_end_date")
                
                force_symbol = cmds.get("force_symbol", "")
                if is_live_paper and force_symbol and force_symbol != "AUTO":
                    args.symbol = force_symbol
        except: pass

    # ── Validar que las fechas sean parseables (evita que "2021-02-31" pase silencioso) ──
    from datetime import datetime as _dt
    def _validate_date(d, label):
        if not d or not d.strip():
            return None
        d = d.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                _dt.strptime(d[:len(fmt.replace('%Y','0000').replace('%m','00').replace('%d','00').replace('%H','00').replace('%M','00').replace('%S','00'))], fmt)
                return d
            except ValueError:
                pass
        # Último intento con pandas
        try:
            import pandas as _pd
            parsed = _pd.to_datetime(d)
            # Verificar que no hubo rollover (ej: Feb 31 → Mar 3)
            reconstructed = parsed.strftime("%Y-%m-%d")
            if d[:10] != reconstructed:
                log.error(f"⛔ Fecha de {label} INVÁLIDA: '{d}' → pandas la interpreta como '{reconstructed}'. Ignorando.")
                return None
            return d
        except Exception as e:
            log.error(f"⛔ Fecha de {label} no parseable: '{d}' → {e}. Ignorando filtro.")
            return None

    sim_start_date = _validate_date(sim_start_date, "INICIO")
    sim_end_date   = _validate_date(sim_end_date,   "FIN")
    
    if sim_start_date:
        log.info(f"📅 Fecha de inicio validada: {sim_start_date}")
    if sim_end_date:
        log.info(f"📅 Fecha de fin validada:    {sim_end_date}")

    # Inyectar start_date y end_date reales

    return HapiMock(
        symbol=args.symbol, 
        initial_cash=args.cash, 
        live_paper=is_live_paper, 
        start_date=sim_start_date,
        end_date=sim_end_date
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
    runner = SimulationRunner(init_broker_callback)
    
    try:
        runner.main_loop(args)
    except SessionInterrupted:
        log.info("📢 Simulación interrumpida por comando global.")
    except KeyboardInterrupt:
        log.info("\n🛑 Proceso detenido por el usuario.")
    except Exception as e:
        log.critical(f"💥 Error fatal en el sistema: {e}", exc_info=True)
    finally:
        log.info("👋 Simulación finalizada.")

if __name__ == "__main__":
    main()
