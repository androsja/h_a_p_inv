import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

from shared import config

log = logging.getLogger("api")

# Los periodos se calculan dinámicamente en _get_curriculum_dates()

class SystemState(BaseModel):
    is_running: bool = False
    performance_profile: str = "turbo" # "eco" o "turbo"
    force_sim_only: bool = False      # 🛡️ Bloqueo global de seguridad
    current_focus: Optional[str] = None
    next_action: Optional[str] = None
    status_log: List[str] = []

class AutoTrainerManager:
    """Gestor autonomo que recorre el pool de activos y los somete a entrenamiento y validacion OOS."""
    
    def __init__(self, docker_mgr):
        self.docker_mgr = docker_mgr
        self.state = SystemState()
        self.assets_path = config.ASSETS_FILE
        self._loop_task: Optional[asyncio.Task] = None
        self.max_concurrent_bots = config.MAX_AUTOTRAINER_BOTS
        self.settings_file = config.DATA_DIR / "auto_trainer_settings.json"
        
        # Inyectado desde el orquestador
        self.mastery_mgr = None
        self.queue_mgr = None
        
        self._load_settings()

    def _load_settings(self):
        """Carga la configuracion persistente."""
        if self.settings_file.exists():
            try:
                data = json.loads(self.settings_file.read_text())
                self.max_concurrent_bots = data.get("max_concurrent_bots", config.MAX_AUTOTRAINER_BOTS)
                self.state.performance_profile = data.get("performance_profile", "turbo")
                self.state.force_sim_only = data.get("force_sim_only", False)
                # Recuperar estado de ejecucion previo
                saved_running = data.get("is_running", False)
                log.info(f"AutoTrainer | Settings cargados: max_bots={self.max_concurrent_bots}, profile={self.state.performance_profile}, force_sim={self.state.force_sim_only}, was_running={saved_running}")
                
                if saved_running:
                    # Usar un pequeño delay para asegurar que el orquestador este listo
                    asyncio.get_event_loop().call_later(2, self.start)
            except Exception as e:
                log.warning(f"AutoTrainer | Error cargando settings: {e}")

    def _save_settings(self):
        """Guarda la configuracion actual."""
        try:
            data = {
                "max_concurrent_bots": self.max_concurrent_bots,
                "performance_profile": self.state.performance_profile,
                "force_sim_only": self.state.force_sim_only,
                "is_running": self.state.is_running
            }
            self.settings_file.write_text(json.dumps(data))
        except Exception as e:
            log.error(f"AutoTrainer | Error guardando settings: {e}")

    def _log(self, msg: str):
        log.info(f"AutoTrainer | {msg}")
        self.state.status_log.insert(0, msg)
        if len(self.state.status_log) > 10:
            self.state.status_log = self.state.status_log[:10]

    def _get_symbols(self) -> List[str]:
        if self.assets_path.exists():
            try:
                data = json.loads(self.assets_path.read_text(encoding="utf-8"))
                assets = data.get("assets", [])
                return [a["symbol"] for a in assets if isinstance(a, dict) and "symbol" in a]
            except:
                pass
        return ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "AMD"] # Fallback

    def _get_curriculum_dates(self, stage: str) -> tuple:
        """Calcula fechas dinámicas (Walk-Forward) relativas al presente."""
        from datetime import datetime, timedelta
        now = datetime.now()
        
        if stage == "SCOUTING":
            # Scouting: Un bloque de 3 meses de hace un año y medio
            end = now - timedelta(days=400)
            start = end - timedelta(days=90)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            
        elif stage == "TRAINING":
            # Entrenamiento: 1 año completo, terminando hace 30 días
            # (Dejamos un mes de margen para evitar Leakage en la validación)
            end = now - timedelta(days=31)
            start = end - timedelta(days=365)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            
        elif stage == "VALIDATING":
            # Validación: Los últimos 30 días hasta hoy (Pura data OOS reciente)
            start = now - timedelta(days=30)
            return start.strftime("%Y-%m-%d"), None # None = hasta hoy
            
        elif stage == "STRESS-TEST":
            # Stress: Los últimos 15 días (Máxima intensidad reciente)
            start = now - timedelta(days=15)
            return start.strftime("%Y-%m-%d"), None
            
        return None, None

    def start(self):
        if not self.state.is_running:
            self.state.is_running = True
            self.state.status_log = []
            self._save_settings()
            self._log("⚡ Auto-Perfeccionador encendido. Iniciando Curriculum...")
            self._loop_task = asyncio.create_task(self._training_loop())

    def stop(self):
        if self.state.is_running:
            self.state.is_running = False
            self._save_settings()
            self._log("🛑 Auto-Perfeccionador apagado.")
            if self._loop_task:
                self._loop_task.cancel()
                self._loop_task = None

    def toggle(self):
        if self.state.is_running:
            self.stop()
        else:
            self.start()
        return self.state.is_running

    def get_status(self) -> Dict:
        data = self.state.model_dump()
        data["max_bots"] = self.max_concurrent_bots # Incluir limite actual
        return data

    def update_settings(self, max_bots: int):
        """Actualiza el límite máximo de bots (Hard-Cap: 10) y persiste."""
        self.max_concurrent_bots = min(10, max(1, max_bots))
        self._save_settings()
        self._log(f"⚙️ Configuración: Límite ajustado a {self.max_concurrent_bots}")

    def update_performance_profile(self, profile: str):
        """Cambia entre eco o turbo (Intensidad)."""
        if profile in ["eco", "turbo"]:
            self.state.performance_profile = profile
            self._save_settings()
            self._log(f"🎭 Perfil de RENDIMIENTO cambiado a: {profile.upper()}")

    def update_safety_mode(self, force_sim: bool):
        """Activa o desactiva el bloqueo de Solo Simulación."""
        self.state.force_sim_only = force_sim
        self._save_settings()
        status = "ACTIVADO (Solo Simulación)" if force_sim else "DESACTIVADO (Permitir Live/Paper)"
        self._log(f"🛡️ Seguro de SEGURIDAD: {status}")

    async def _training_loop(self):
        self._log("Bucle de entrenamiento iniciado.")
        while self.state.is_running:
            try:
                await self._process_tick()
            except asyncio.CancelledError:
                self._log("Bucle interno cancelado por el sistema.")
                break
            except Exception as e:
                self._log(f"⚠️ Advertencia en el tick: {e}. Reintentando en 10s...")
            
            await asyncio.sleep(10) # Evalua cada 10 segundos

    async def _process_tick(self):
        active_bots = self.docker_mgr.list_active_bots()
        
        # 🎭 APLICAR PERFIL DE RENDIMIENTO
        if self.state.performance_profile == "eco":
            target_limit = 2 # En modo trabajo solo permitimos 2 bots para no molestar
            reason = " (🍃 Modo Eco-Trabajo activo)"
        else:
            # Cuántos nuevos slots adicionales puede soportar la RAM / CPU (respetando el tope de max_concurrent_bots)
            needed_slots = max(0, self.max_concurrent_bots - len(active_bots))
            new_safe_slots = self.docker_mgr.get_available_slots(base_limit=needed_slots)
            
            target_limit = len(active_bots) + new_safe_slots
            
            health = self.docker_mgr.get_resource_health()
            ram_avail = health.get('ram_avail_mb', 0)
            reason = ""
            ram_avail = health.get('ram_avail_mb', 0)
            reason = ""
            if self.state.force_sim_only:
                reason = " (🛡️ Modo Simulación Forzada)"
            elif target_limit < self.max_concurrent_bots:
                reason = f" (🚀 Limitado por RAM: {ram_avail}MB disponibles)"

        self.state.next_action = f"Vigilando {len(active_bots)}/{target_limit} bots{reason}"
        
        if target_limit <= 0 and self.state.performance_profile == "turbo":
            self.state.current_focus = "⚠️ PAUSADO: RAM INSUFICIENTE"
            return

        # --- SINCRONIZACION DE MAESTRIA CON RESULTADOS ---
        self._sync_mastery_from_results()

        # ─── PRIORIDAD DE LANZAMIENTO ─────────────────────────────────────────
        # Construir el ranking global (activos + pendientes) para detectar si hay
        # algún candidato de mayor prioridad esperando mientras slots están ocupados
        # por bots de prioridad baja (SCOUTING). Si es así, desalojar al más débil.
        def sort_priority(item):
            if item["status"] in ["ELITE", "READY_FOR_LIVE"]: return 100 + item["rank"]
            if item["status"] == "LEARNING": return 50 + item["rank"]
            return item["rank"]

        all_symbols = self._get_symbols()
        active_symbols = set(b.get("symbol") for b in active_bots)
        completed_bots = self.docker_mgr.list_completed_bots()
        latest_history = {}
        for b in completed_bots:
            sym = b.get("symbol")
            if sym not in latest_history:
                latest_history[sym] = b

        # Calcular prioridad de todos los símbolos pendientes (no activos)
        pending = []
        for sym in all_symbols:
            if sym in active_symbols:
                continue
            d = self.mastery_mgr.get_symbol_status(sym) if self.mastery_mgr else {}
            pending.append({"symbol": sym, "rank": d.get("rank", 0), "status": d.get("status", "UNKNOWN")})
        pending.sort(key=sort_priority, reverse=True)

        # Si hay candidatos de alta prioridad esperando y los slots están llenos con SCOUTING:
        # → Desalojar el bot activo de MENOR prioridad para dar paso al de MAYOR prioridad
        if len(active_bots) >= target_limit and pending:
            best_pending_pri = sort_priority(pending[0]) if pending else 0
            # Calcular prioridad de cada bot activo
            active_with_pri = []
            for b in active_bots:
                sym = b.get("symbol", "")
                d = self.mastery_mgr.get_symbol_status(sym) if self.mastery_mgr else {}
                pri = sort_priority({"status": d.get("status", "UNKNOWN"), "rank": d.get("rank", 0)})
                active_with_pri.append((sym, pri, b))
            # El bot activo con menor prioridad
            if active_with_pri:
                lowest_sym, lowest_pri, _ = min(active_with_pri, key=lambda x: x[1])
                if best_pending_pri > lowest_pri + 10:  # Solo desalojar si la diferencia es significativa
                    self._log(f"🔄 Desalojando {lowest_sym} (pri={lowest_pri:.0f}) → lugar para {pending[0]['symbol']} (pri={best_pending_pri:.0f})")
                    self.docker_mgr.kill_bot(lowest_sym)
                    active_bots = self.docker_mgr.list_active_bots()
                    active_symbols = set(b.get("symbol") for b in active_bots)

        # 🚀 LANZAMIENTO EN RAFAGA: Intentamos llenar todos los slots en este mismo tick
        while len(active_bots) < target_limit:
            target_to_launch = None
            target_mode = None
            start_dt = None
            end_dt = None
            freeze = False
            is_from_queue = False

            # --- PRIORIDAD 1: COLA MANUAL DEL USUARIO ---
            if self.queue_mgr:
                queued = self.queue_mgr.get_queue()
                if queued:
                    next_item = self.queue_mgr.pop_next()
                    if next_item:
                        target_to_launch = next_item.get("symbol")
                        target_mode = next_item.get("stage", "TRAINING")
                        start_dt = next_item.get("start_date")
                        end_dt = next_item.get("end_date")
                        freeze = next_item.get("freeze_learning", False)
                        launched_mode = next_item.get("mode", "SIMULATED")
                        is_from_queue = True
                        
                        self._log(f"📥 Procesando elemento de COLA: {target_to_launch} ({target_mode})")
            
            # --- PRIORIDAD 2: CURRICULUM AUTOMATICO (Si no hubo nada en cola) ---
            if not is_from_queue:
                active_symbols = set(b.get("symbol") for b in active_bots)
                completed_bots = self.docker_mgr.list_completed_bots()
                latest_history = {}
                for b in completed_bots:
                    sym = b.get("symbol")
                    if sym not in latest_history:
                        latest_history[sym] = b

                # 1. Obtener candidatos ordenados por rango de maestria
                symbol_ranks = []
                for sym in all_symbols:
                    if sym in active_symbols: continue
                    status_data = self.mastery_mgr.get_symbol_status(sym)
                    symbol_ranks.append({
                        "symbol": sym,
                        "rank": status_data.get("rank", 0),
                        "status": status_data.get("status", "LEARNING")
                    })

                symbol_ranks.sort(key=sort_priority, reverse=True)

                # 2. Buscar el siguiente para lanzar
                for item in symbol_ranks:
                    sym = item["symbol"]
                    
                    # ── FUENTE DE VERDAD: pvc_stage del MasteryHub ──────────────────
                    mastery_data = self.mastery_mgr.get_symbol_status(sym) if self.mastery_mgr else {}
                    pvc_stage = mastery_data.get("pvc_stage", "SCOUTING")
                    history = latest_history.get(sym)
                    last_stage = history.get("stage", "") if history else ""

                    # 🛡️ FILTRO DE DIVERSIFICACIÓN (SOLO PARA TRADING REAL/PAPER)
                    # En simulación NO tiene sentido limitar por sector; queremos validar lo mejor rápido.
                    
                    # Solo aplicar límite si el modo final será Live/Shadow
                    if pvc_stage == "LIVE-SHADOW":
                        from shared.utils.metadata import get_sector_for
                        sector = get_sector_for(sym)
                        active_sectors = [get_sector_for(b["symbol"]) for b in active_bots]
                        if active_sectors.count(sector) >= 2:
                            continue # Saltar a otro sector para diversificar en Shadow/Live

                    # La fase a lanzar la dicta pvc_stage. Solo se evita re-lanzar la misma fase.
                    if pvc_stage == "SCOUTING":
                        if last_stage != "SCOUTING":
                            target_to_launch = sym
                            target_mode = "SCOUTING"
                            start_dt, end_dt = self._get_curriculum_dates("SCOUTING")
                            break
                        continue

                    elif pvc_stage == "TRAINING":
                        if last_stage != "TRAINING":
                            target_to_launch = sym
                            target_mode = "TRAINING"
                            start_dt, end_dt = self._get_curriculum_dates("TRAINING")
                            break
                        continue

                    elif pvc_stage == "VALIDATING":
                        if last_stage != "VALIDATING":
                            target_to_launch = sym
                            target_mode = "VALIDATING"
                            start_dt, end_dt = self._get_curriculum_dates("VALIDATING")
                            break
                        continue

                    elif pvc_stage == "STRESS-TEST":
                        if last_stage != "STRESS-TEST":
                            target_to_launch = sym
                            target_mode = "STRESS-TEST"
                            start_dt, end_dt = self._get_curriculum_dates("STRESS-TEST")
                            break
                        continue

                    elif pvc_stage == "LIVE-SHADOW":
                        if last_stage != "LIVE-SHADOW":
                            target_to_launch = sym
                            target_mode = "LIVE-SHADOW"
                            start_dt, end_dt = None, None
                            break
                        continue

            if target_to_launch:
                if not is_from_queue:
                    # 🔒 Determinar si esta fase debe congelar el aprendizaje (Solo para curriculum)
                    EVALUATION_PHASES = {"VALIDATING", "STRESS-TEST", "LIVE-SHADOW"}
                    freeze = target_mode in EVALUATION_PHASES
                    
                    # 🛡️ SOBREESCRITURA PARA MODO SIM-ONLY (SEGURIDAD)
                    if self.state.force_sim_only:
                        launched_mode = "SIMULATED"
                        if target_mode == "LIVE-SHADOW":
                            # En modo sim-only, las de LIVE-SHADOW se corren como una validación de 30 días
                            start_dt, end_dt = self._get_curriculum_dates("VALIDATING")
                    else:
                        launched_mode = "LIVE_PAPER" if target_mode == "LIVE-SHADOW" else "SIMULATED"
                
                # 🏷️ ETIQUETA DE MODO PARA EL LOG
                display_mode = target_mode
                if self.state.force_sim_only and target_mode == "LIVE-SHADOW":
                    display_mode = "SIMULACIÓN-FORZADA 🛡️"
                
                freeze_icon = "🔒 Frozen" if freeze else "🧠 Learning"
                source_tip = "[COLA]" if is_from_queue else "[AUTO]"
                
                # 📢 REGISTRO DETALLADO DEL MOTIVO (Para que el usuario entienda "porque iniciaron")
                mastery_data = self.mastery_mgr.get_symbol_status(target_to_launch) if self.mastery_mgr else {}
                rank = mastery_data.get("rank", 0)
                status = mastery_data.get("status", "SCOUTING")
                
                log_msg = f"🚀 {source_tip} Lanzando {target_to_launch} en fase {display_mode} [{freeze_icon}]"
                if not is_from_queue:
                    log_msg += f" - Motivo: Rango {rank} alcanzó estatus {status}"
                
                self._log(log_msg)
                
                success, msg = self.docker_mgr.launch_bot(
                    symbol=target_to_launch, 
                    mode=launched_mode, 
                    start_date=start_dt, 
                    end_date=end_dt,
                    freeze_learning=freeze,   # 🔒 Congelar pesos en evaluación
                    stage=target_mode,        # 🏷️ Etiquetar la fase en el histórico
                )
                
                if success:
                    active_bots.append({"symbol": target_to_launch.upper()})
                    self.state.current_focus = f"Scalado: {len(active_bots)} bots"
                else:
                    self._log(f"❌ Error lanzando {target_to_launch}: {msg}")
                    break 
            else:
                self.state.current_focus = "Ranking Completo"
                self.state.next_action = "Todos los candidatos estan operando o validados."
                break 

    def _sync_mastery_from_results(self):
        """Lee backtest_results.json y actualiza el MasteryHub."""
        results_file = config.DATA_DIR / "backtest_results.json"
        if not results_file.exists():
            return
            
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
                
            for session in data:
                symbol = session.get("symbol")
                if symbol:
                    self.mastery_mgr.update_symbol_metrics(symbol, session)
        except Exception as e:
            log.warning(f"AutoTrainer | Error sincronizando maestria: {e}")
