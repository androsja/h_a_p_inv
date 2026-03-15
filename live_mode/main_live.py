"""
main_live.py ─ Motor principal del Bot de Trading Automatizado (Modo Real / Live Paper).
"""

import sys
import argparse
import os
import json
from shared import config
from shared.utils.logger import log
from shared.broker.interface import BrokerInterface
from shared.broker.hapi_live import HapiLive
from shared.broker.hapi_mock import HapiMock
from shared.utils.market_hours import market_status_str
from shared.engine.runner import SimulationRunner
from shared.engine.trading_engine import TradingEngine

BANNER = """
╔══════════════════════════════════════════════════════╗
║     🇨🇴 HAPI SCALPING BOT  ─  Colombia Edition       ║
║     Estrategia: EMA Crossover + RSI Filter           ║
║     Modo: LIVE / LIVE PAPER TRADING (Real Delta)     ║
╚══════════════════════════════════════════════════════╝
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot de Trading - Modo LIVE")
    parser.add_argument("--symbol", default=None, help="Símbolo específico para operar.")
    parser.add_argument("--paper", action="store_true", help="Activar modo PAPER TRADING con datos en vivo.")
    parser.add_argument("--cash", type=float, default=10000.0, help="Capital inicial para Paper ($)")
    
    p = parser.parse_args()
    p.mode = "LIVE"
    
    fixed = os.getenv("FIXED_SYMBOL", "").strip()
    if fixed and not p.symbol:
        p.symbol = fixed
    return p

def init_broker_callback(args: argparse.Namespace) -> BrokerInterface:
    """Inicializa el bróker basado en si es Paper o Real Live."""
    # Detectar si hay anulación de Live Paper desde comandos globales
    is_lp_override = False
    try:
        cmd_file = config.COMMAND_FILE
        if cmd_file.exists():
            with open(cmd_file) as f:
                cmds = json.load(f)
            is_lp_override = cmds.get("force_paper_trading", False)
    except: pass

    if args.paper or is_lp_override:
        log.info("🧪 Bróker: HapiMock configurado para LIVE PAPER (Datos reales)")
        return HapiMock(symbol=args.symbol, initial_cash=args.cash, live_paper=True)
        
    # Verificar credenciales en .env para Real Live
    api_key    = config.HAPI_API_KEY
    client_id  = config.HAPI_CLIENT_ID
    user_token = config.HAPI_USER_TOKEN

    if not all([api_key, client_id, user_token]):
        log.error("❌ ERROR CRÍTICO: Credenciales de Hapi no encontradas en .env.")
        sys.exit(1)

    log.info("💰 Bróker: HapiLive conectado (CUENTA REAL)")
    return HapiLive(api_key=api_key, client_id=client_id, user_token=user_token)

def run_bot_compat(broker, args, session_num=1, stop_event=None, asset_type="normal"):
    """Wrapper para el motor de ejecución."""
    engine = TradingEngine(broker, args)
    engine.run(session_num=session_num, stop_event=stop_event, asset_type=asset_type)

def main():
    print(BANNER)
    args = parse_args()
    
    # Configuración de archivos de estado
    from shared.utils.state_writer import set_state_file
    from shared.data.market_data import set_assets_file
    set_state_file(config.STATE_FILE_LIVE)
    set_assets_file(config.ASSETS_FILE_LIVE)
    
    log.info(f"🚀 Iniciando en modo LIVE | {'PAPER' if args.paper else 'REAL'}")
    log.info(market_status_str())

    # El Runner orquestará la ejecución
    runner = SimulationRunner(init_broker_callback, run_bot_compat)
    
    try:
        runner.main_loop(args)
    except KeyboardInterrupt:
        log.info("\n🛑 Proceso LIVE detenido por el usuario.")
    except Exception as e:
        log.critical(f"💥 Error fatal en modo LIVE: {e}", exc_info=True)
    finally:
        log.info("👋 Sesión LIVE finalizada.")

if __name__ == "__main__":
    main()
