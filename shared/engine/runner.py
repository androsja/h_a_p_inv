"""
shared/engine/runner.py ─ Orquestador de trading compartido (SIM/LIVE).
"""

import os
import json
import importlib
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from shared import config
from shared.utils.logger import log
from shared.data import market_data
from shared.utils.state_writer import update_state, clear_state
from shared.utils.market_hours import _is_mock_time_active
from shared.utils.checkpoint import load_simulation_checkpoint, save_simulation_checkpoint, clear_simulation_checkpoints
from shared.utils.neural_filter import get_neural_filter
from shared.utils.live_paper_launcher import launch_parallel_bots

from .utils import SessionInterrupted, smart_sleep
from .trading_engine import TradingEngine

class SimulationRunner:
    """
    Gestiona la ejecución secuencial o paralela de múltiples sesiones de trading.
    """
    def __init__(self, init_broker_fn, run_bot_fn):
        self.init_broker = init_broker_fn
        self.run_bot_logic = run_bot_fn # Para compatibilidad con launch_parallel_bots
        self.session_num = 0
        self.symbol_idx = 0
        self.all_symbols = []
        self.all_assets = []
        self.is_simulated = True
        self.session_pause = config.SESSION_PAUSE
        
        # 📈 Acumuladores Globales de Simulación
        self.total_sim_trades = 0
        self.total_sim_wins = 0
        self.total_sim_pnl = 0.0
        self.total_sim_ghosts = 0
        self.all_done = False  # 🏁 Bandera de finalización global
        self.start_real_time = None # 🕒 Momento real en que inició la simulación global
        
    def main_loop(self, args: argparse.Namespace):
        log.info("🚀 SISTEMA INICIADO: Preparando motores de trading...")
        self.is_simulated = (args.mode == "SIMULATED")
        
        # Restore checkpoint
        self._restore_checkpoint()
        
        while True:
            try:
                # 1. Comandos Globales (Prioridad Máxima - Procesa reinicios)
                if self._process_global_commands():
                    continue # Se reinició el estado, volver al inicio
                
                # Registrar inicio real si es el primer símbolo
                if self.is_simulated and self.symbol_idx == 0 and not self.start_real_time:
                    self.start_real_time = datetime.now()
                
                if self.all_done:
                    smart_sleep(3) # Idle mientras esperamos nuevos comandos (interrumpible)
                    continue

                # 2. Recarga Dinámica
                self._reload_symbols()
                
                # 3. Fin de lista?
                if self.is_simulated and self.all_symbols and self.symbol_idx >= len(self.all_symbols):
                    self._handle_completion()
                    continue # Ir al inicio del bucle donde se chequea self.all_done
                
                # 4. Siguiente Sesión
                if self.is_simulated and self.all_symbols:
                    args.symbol = self.all_symbols[self.symbol_idx]
                
                self.session_num += 1
                
                # 5. Live Paper Parallel?
                if self._check_live_paper(args):
                    continue
                    
                # 6. Ejecución Serial (Simulación)
                self._run_single_session(args)

                if not self.is_simulated: break
                
                # 7. Siguiente Activo
                self._advance_to_next()
                smart_sleep(self.session_pause)

            except SessionInterrupted:
                log.info("📢 Interrupción detectada (Reinicio/Cambio de modo). Volviendo al inicio del bucle...")
                continue
            except Exception as e:
                log.error(f"❌ Error crítico en bucle principal: {e}", exc_info=True)
                smart_sleep(5) # Evitar bucle infinito de errores rápidos

    def _restore_checkpoint(self):
        try:
            ckpt = load_simulation_checkpoint()
            if self.is_simulated:
                self.symbol_idx = ckpt.get("symbol_idx", 0)
                self.session_num = ckpt.get("session_num", 0)
                if ckpt.get("is_finished"):
                    self.all_done = True
                    log.info("🏁 Checkpoint indica SIMULACIÓN YA COMPLETADA.")
                elif self.symbol_idx > 0:
                    log.info(f"💾 CHECKPOINT restaurado: reanudando desde {ckpt.get('symbol','─')} (idx={self.symbol_idx})")
        except: pass

    def _reload_symbols(self):
        try:
            importlib.reload(market_data)
            from shared.data.market_data import set_assets_file
            set_assets_file(config.ASSETS_FILE_SIM)
            with open(config.ASSETS_FILE_SIM, "r") as f:
                data = json.load(f)
                self.all_assets = [a for a in data.get("assets", []) if a.get("enabled", True)]
                self.all_symbols = [a["symbol"] for a in self.all_assets]
                
                # Report total to dashboard
                update_state(
                    "─", # Positional
                    mode="SIMULATED", status="starting"
                )
                # Solo loguear si es la primera vez o si cambió la lista
                if not getattr(self, '_last_symbols_count', None) == len(self.all_symbols):
                    log.info(f"📋 Cargadas {len(self.all_symbols)} empresas para simular.")
                    self._last_symbols_count = len(self.all_symbols)
        except Exception as e:
            log.warning(f"⚠️ Error recargando símbolos: {e}")

    def _process_global_commands(self):
        cmd_file = config.COMMAND_FILE
        if not cmd_file.exists(): return False
        
        try:
            with open(cmd_file, "r") as f: cmds = json.load(f)
            
            # Snapshot reload / Freeze IA
            self._sync_ia_status(cmds)
            
            if cmds.get("reset_all") or cmds.get("restart_sim"):
                is_purgue = cmds.get("reset_all", False)
                log.info(f"🔄 {'PURGANDO' if is_purgue else 'REINICIANDO'} simulación...")
                
                cmds["reset_all"] = False
                cmds["restart_sim"] = False
                with open(cmd_file, "w") as f: json.dump(cmds, f)
                
                # 🔄 RECARGA DINÁMICA DE CÓDIGO (Estrategia e IA)
                try:
                    import shared.strategy.indicators
                    import shared.strategy.ml_predictor
                    import shared.utils.neural_filter
                    import shared.engine.trading_engine
                    importlib.reload(shared.strategy.indicators)
                    importlib.reload(shared.strategy.ml_predictor)
                    importlib.reload(shared.utils.neural_filter)
                    importlib.reload(shared.engine.trading_engine)
                    log.info("✅ Módulos de estrategia e IA recargados satisfactoriamente.")
                except Exception as e:
                    log.warning(f"⚠️ Error recargando módulos de código: {e}")

                clear_state()
                self._cleanup_files(is_purgue)
                
                self.symbol_idx = 0
                self.session_num = 0
                self.total_sim_trades = 0
                self.total_sim_wins = 0
                self.total_sim_pnl = 0.0
                self.total_sim_ghosts = 0
                self.all_done = False # Reset bandera al reiniciar manualmente
                self.start_real_time = None # Reset tiempo real
                update_state("─", mode="SIMULATED", status="restarting", mock_time_930=_is_mock_time_active())
                return True
        except: pass
        return False

    def _sync_ia_status(self, cmds):
        frozen = cmds.get("strategy_frozen", False)
        # Sincronizar NeuralFilter — aplicar freeze/unfreeze a todos los modelos registrados
        try:
            from shared.utils.neural_filter import _filter_registry
            for nf in list(_filter_registry.values()):
                if frozen and not nf.is_frozen: nf.freeze()
                elif not frozen and nf.is_frozen: nf.unfreeze()
        except: pass

        if cmds.get("reload_models"):
            cmds["reload_models"] = False
            with open(config.COMMAND_FILE, "w") as f: json.dump(cmds, f)

    def _cleanup_files(self, is_purgue):
        for f in [config.RESULTS_FILE, config.TRADE_JOURNAL_FILE]:
            if f.exists(): f.unlink()
        if is_purgue:
            clear_simulation_checkpoints()
            # Borrar modelos y dataset de IA en memoria y disco
            try:
                from shared.utils.neural_filter import reset_neural_filter
                reset_neural_filter()
            except Exception as e:
                log.error(f"Error reseteando Neural Filter: {e}")
            if config.NEURAL_MODEL_FILE.exists(): config.NEURAL_MODEL_FILE.unlink()
            if config.ML_DATASET_FILE.exists(): config.ML_DATASET_FILE.unlink()

    def _handle_completion(self):
        """Se ejecuta cuando todos los símbolos de la lista han sido procesados."""
        if getattr(self, "all_done", False): return True # Ya reportado anteriormente
        
        log.info("🏁 SIMULACIÓN GLOBAL FINALIZADA. Guardando resultados en el historial...")
        self._save_full_run_to_history()
        
        self.all_done = True
        # PERSISTIR el hecho de que ya terminamos para evitar bucles tras reinicio
        from shared.utils import checkpoint
        checkpoint.save_simulation_checkpoint(self.symbol_idx, "FIN", self.session_num, is_finished=True)
        
        # Reportar estado final con TOTALES para que no se vea en cero
        wr = (self.total_sim_wins / self.total_sim_trades * 100) if getattr(self, "total_sim_trades", 0) > 0 else 0
        update_state(
            mode="SIMULATED", 
            status="completed", 
            symbol="─", 
            session=self.session_num,
            total_sim_trades=int(getattr(self, "total_sim_trades", 0)),
            total_sim_wins=int(getattr(self, "total_sim_wins", 0)),
            total_sim_pnl=float(round(float(getattr(self, "total_sim_pnl", 0.0)), 2)),
            total_sim_ghosts=int(getattr(self, "total_sim_ghosts", 0)),
            win_rate=float(round(float(wr), 2))
        )
        return True

    def _save_full_run_to_history(self):
        """Persiste el resumen de la simulación actual en sim_history.json para Analytics."""
        try:
            from shared.utils.neural_filter import get_neural_filter, _filter_registry
            # Reportar accuracy promedio de todos los modelos por símbolo activos
            all_stats = [nf.get_stats() for nf in _filter_registry.values() if nf.get_stats().get('total_samples', 0) > 0]
            accuracy = sum(s.get('model_accuracy', 0.0) for s in all_stats) / len(all_stats) if all_stats else 0.0
            
            # Leer resultados detallados (símbolo por símbolo) del backtest actual
            detailed_results = []
            if config.RESULTS_FILE.exists():
                with open(config.RESULTS_FILE, "r") as f:
                    detailed_results = json.load(f)

            history_file = config.DATA_CACHE_DIR / "sim_history.json"
            config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            history = []
            if history_file.exists():
                with open(history_file, "r") as f:
                    history = json.load(f)

            win_rate = (getattr(self, "total_sim_wins", 0) / getattr(self, "total_sim_trades", 1) * 100) if getattr(self, "total_sim_trades", 0) > 0 else 0
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "symbols_count": len(self.all_symbols),
                "trades_learned": getattr(self, "total_sim_trades", 0),
                "total_ghosts": getattr(self, "total_sim_ghosts", 0),
                "win_rate": round(win_rate, 2),
                "accuracy": round(accuracy, 2),
                "pnl": round(getattr(self, "total_sim_pnl", 0.0), 2),
                "symbols_list": self.all_symbols,
                "detailed_results": detailed_results,
                "sim_start": detailed_results[0].get("sim_start", "─") if detailed_results else "─",
                "sim_end": detailed_results[-1].get("sim_end", "─") if detailed_results else "─",
                "start_real_time": self.start_real_time.isoformat() if self.start_real_time else None,
                "investment_style": "Normal" # Opcional: detectar del engine
            }

            # Calcular días simulados si tenemos fechas válidas
            if entry["sim_start"] != "─" and entry["sim_end"] != "─":
                try:
                    d1 = datetime.strptime(entry["sim_start"][:10], "%Y-%m-%d")
                    d2 = datetime.strptime(entry["sim_end"][:10], "%Y-%m-%d")
                    entry["sim_days"] = (d2 - d1).days + 1
                except:
                    entry["sim_days"] = 0
            else:
                entry["sim_days"] = 0

            history.append(entry)
            history = history[-500:] # Mantener últimas 500
            
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
            
            log.info(f"✅ Historial actualizado: {len(self.all_symbols)} activos, PnL: ${entry['pnl']}")
        except Exception as e:
            log.error(f"❌ Error guardando historial en backend: {e}")

    def _check_live_paper(self, args):
        is_lp = False
        force_symbols = []
        try:
            if config.COMMAND_FILE.exists():
                with open(config.COMMAND_FILE) as f:
                    c = json.load(f)
                    is_lp = c.get("force_paper_trading", False)
                    force_symbols = c.get("force_symbols", [])
        except: pass
        
        if is_lp and force_symbols:
            log.info(f"🚀 Modo Live Paper detectado ({len(force_symbols)} símbolos).")
            try:
                launch_parallel_bots(args, force_symbols, self.session_num, self.run_bot_logic, self.init_broker, save_simulation_checkpoint)
            except SessionInterrupted:
                log.info("📢 Live Paper interrumpido por comando global.")
            return True
        return False

    def _run_single_session(self, args):
        broker = self.init_broker(args)
        asset_type = "normal"
        if self.all_assets and self.symbol_idx < len(self.all_assets):
            asset_type = self.all_assets[self.symbol_idx].get("type", "normal")
            
        engine = TradingEngine(broker, args)
        engine.run(session_num=self.session_num, asset_type=asset_type)
        
        # 8. Record Results
        self._record_session_results(engine, broker)

    def _record_session_results(self, engine, broker):
        try:
            stats = broker.stats
            # Only record if it's simulated and has some data
            if not self.is_simulated: return
            
            pnl_val = getattr(stats, 'total_pnl', 0.0)
            trades_val = getattr(stats, 'total_trades', 0)
            winrate_val = getattr(stats, 'win_rate', 0.0)
            
            insight = "Sin actividad."
            if trades_val == 0:
                insight = "No hubo entradas. Filtro de calidad o condiciones estrictas."
            elif pnl_val > 0:
                insight = "ESTRATEGIA EXITOSA. Rentabilidad positiva en este activo."
            else:
                insight = "ESTRATEGIA FALLIDA. Pérdidas en este escenario."

            from datetime import datetime, timezone
            total_fees = round(getattr(stats, 'total_fees', 0.0), 4)

            # Get IA accuracy for the current symbol specifically
            accuracy_ia = 0.0
            total_samples = 0
            try:
                from shared.utils.neural_filter import _filter_registry
                if engine.symbol in _filter_registry:
                    nft_stats = _filter_registry[engine.symbol].get_stats()
                    accuracy_ia = nft_stats.get('model_accuracy', 0.0)
                    total_samples = nft_stats.get('total_samples', 0)
            except: pass
            
            res_file = config.RESULTS_FILE
            
            # Extract last trade data from broker history
            last_price = None
            last_order_date = None
            try:
                broker_trades = getattr(stats, 'trade_history', None) or getattr(broker, 'trade_history', None) or []
                if broker_trades:
                    last_t = broker_trades[-1]
                    last_price = last_t.get('price') or last_t.get('fill_price') or last_t.get('exit_price')
                    last_order_date = last_t.get('timestamp') or last_t.get('timestamp_close') or datetime.now(timezone.utc).isoformat()
            except Exception:
                pass

            session_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": engine.symbol,
                "session_num": self.session_num,
                "total_trades": trades_val,
                "winning_trades": getattr(stats, 'winning_trades', 0),
                "losing_trades": trades_val - getattr(stats, 'winning_trades', 0),
                "win_rate": round(winrate_val, 2),
                "accuracy": accuracy_ia,
                "total_samples": total_samples,
                "pnl": round(pnl_val, 2),
                "gross_pnl": round(pnl_val + total_fees, 2),
                "total_fees": total_fees,
                "slippage_est": round(trades_val * 0.10, 2),
                "gross_profit": round(getattr(stats, 'gross_profit', 0.0), 2),
                "gross_loss": round(getattr(stats, 'gross_loss', 0.0), 2),
                "drawdown": round(getattr(stats, 'max_drawdown', 0.0), 2),
                "insight": insight,
                "sim_start": engine.sim_start_date,
                "sim_end": engine.sim_end_date,
                "sim_duration": round(time.time() - getattr(engine, '_engine_start_time', time.time()), 2),
                "investment_style": engine.investment_style,
                "blocking_summary": dict(engine.blocking_history or {}),
                "ghost_trades_count": len(getattr(engine, 'ghost_history', []) or []) + len(getattr(engine, 'ghost_positions', []) or []),
                "total_ghosts": len(getattr(engine, 'ghost_history', []) or []) + len(getattr(engine, 'ghost_positions', []) or []),
                "last_price": round(float(last_price), 2) if last_price else None,
                "last_order_date": last_order_date,
            }

            # Accumulate global totals
            self.total_sim_trades += int(trades_val)
            self.total_sim_wins += int(getattr(stats, 'winning_trades', 0))
            self.total_sim_pnl += float(pnl_val)
            self.total_sim_ghosts += int(session_result.get("total_ghosts", 0))

            # Update dashboard with global report
            # Excluir 'accuracy' del kwargs (el parámetro correcto es model_accuracy)
            excluded_keys = ["symbol", "status", "blocking_summary", "timestamp", "session_num", "accuracy", "total_samples"]
            update_state(
                symbol=engine.symbol,
                status="completed",
                session=session_result.get("session_num", self.session_num),
                blocking_summary=session_result.get("blocking_summary"),
                model_accuracy=accuracy_ia,   # ← parámetro correcto para state_writer
                total_samples=total_samples,  # ← parámetro correcto para state_writer
                **{k:v for k,v in session_result.items() if k not in excluded_keys}
            )

            update_state(
                "─", 
                mode="SIMULATED", 
                total_sim_trades=self.total_sim_trades,
                total_sim_wins=self.total_sim_wins,
                total_sim_pnl=round(self.total_sim_pnl, 2),
                total_sim_ghosts=self.total_sim_ghosts,
                report=session_result
            )

            results = []
            if res_file.exists():
                with open(res_file, "r") as f:
                    try: results = json.load(f)
                    except: results = []
            
            # De-duplicate
            results = [r for r in results if not (r.get("symbol") == engine.symbol and r.get("session_num") == self.session_num)]
            results.append(session_result)
            
            with open(res_file, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            log.warning(f"Error recording session results: {e}")

    def _advance_to_next(self):
        if self.is_simulated and self.all_symbols:
            self.symbol_idx += 1
            save_simulation_checkpoint(
                self.symbol_idx, 
                self.all_symbols[self.symbol_idx] if self.symbol_idx < len(self.all_symbols) else "─",
                self.session_num
            )
            if self.symbol_idx < len(self.all_symbols):
                log.info(f"⏸ Sesión finalizada. Siguiente: {self.all_symbols[self.symbol_idx]}...")
