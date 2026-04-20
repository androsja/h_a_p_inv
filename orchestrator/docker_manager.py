import os
import docker
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import csv
from shared import config

COMPLETED_LOG = config.DATA_DIR / "completed_simulations.json"
ACTIVE_SESSIONS_LOG = config.DATA_DIR / "active_sessions.json"

class DockerManager:
    def __init__(self):
        self.is_connected = False
        self.client = None
        self._connect_docker()
        self.journal_path = config.TRADE_JOURNAL_FILE
        
        # --- Configuración Realista (Medido: ~250MB/bot) ---
        self._RAM_PER_SIM_MB  = 300
        self._CPU_PER_SIM_PCT = 15
        self._MIN_FREE_RAM_MB = config.MIN_FREE_RAM_MB

        # Registro persistente de bots completados (cargado desde disco)
        self.completed_bots: List[Dict] = self._load_completed()

        # Metadatos de lanzamientos activos (Persistido en disco)
        self._launch_meta: Dict[str, Dict] = self._load_active_sessions()
        
    def clear_memory(self):
        """Limpia el registro de bots en memoria y sesiones activas."""
        self.completed_bots = []
        self._launch_meta = {}
        # También guardar en disco para que el archivo quede vacío
        self._save_completed()
        self._save_active_sessions()
        print("🧹 [DockerManager] Memoria de bots y sesiones limpiada.")

    def _connect_docker(self):
        """Intenta (re)conectar con el daemon de Docker con timeout."""
        try:
            # En Mac, usamos un timeout corto para evitar que el orquestador se cuelgue si el daemon no responde
            self.client = docker.from_env(timeout=5)
            # Test de conexión rápido
            self.client.ping()
            self.is_connected = True
        except Exception as e:
            print(f"⚠️ [DockerManager] No se pudo conectar a Docker: {e}")
            self.is_connected = False
            self.client = None

    def _load_completed(self) -> List[Dict]:
        """Carga el historial de simulaciones completadas desde disco."""
        try:
            if COMPLETED_LOG.exists():
                data = json.loads(COMPLETED_LOG.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # 🔥 Migración de diccionario a lista
                    migrated = []
                    for k, v in data.items():
                        v["symbol"] = k
                        v["run_id"] = f"{k}_{v.get('finished_at', 'old').replace(' ', '_')}"
                        migrated.append(v)
                    return migrated
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[DockerManager] No se pudo cargar completed log: {e}")
        return []

    def _save_completed(self):
        """Persiste el historial en disco de forma atómica."""
        try:
            COMPLETED_LOG.parent.mkdir(parents=True, exist_ok=True)
            temp_file = COMPLETED_LOG.with_suffix(".tmp")
            temp_file.write_text(json.dumps(self.completed_bots, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_file, COMPLETED_LOG) # Operación atómica en Unix
        except Exception as e:
            print(f"[DockerManager] No se pudo guardar completed log: {e}")

    def _load_active_sessions(self) -> Dict:
        """Carga los metadatos de sesiones activas desde disco y limpia fantasmas."""
        try:
            if ACTIVE_SESSIONS_LOG.exists():
                data = json.loads(ACTIVE_SESSIONS_LOG.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    return {}
                # 🧹 Limpiar sesiones fantasma: verificar que cada símbolo tenga un contenedor real
                if self.is_connected and self.client:
                    try:
                        running = {c.name for c in self.client.containers.list()}
                        clean = {}
                        for sym, meta in data.items():
                            mode = meta.get("mode", "SIMULATED").lower()
                            expected_name = f"hapi_worker_{mode}_{sym.upper()}"
                            if expected_name in running:
                                clean[sym] = meta
                            else:
                                print(f"🧹 [DockerManager] Sesión fantasma eliminada: {sym} (contenedor '{expected_name}' no existe)")
                        if len(clean) < len(data):
                            # Persistir la versión limpia
                            self._launch_meta = clean
                            self._save_active_sessions()
                        return clean
                    except Exception as e:
                        print(f"[DockerManager] No se pudo verificar contenedores al cargar sesiones: {e}")
                return data
        except Exception as e:
            print(f"[DockerManager] No se pudo cargar active sessions: {e}")
        return {}

    def _save_active_sessions(self):
        """Guarda los metadatos de sesiones activas de forma atómica."""
        try:
            ACTIVE_SESSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
            temp_file = ACTIVE_SESSIONS_LOG.with_suffix(".tmp")
            temp_file.write_text(json.dumps(self._launch_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_file, ACTIVE_SESSIONS_LOG)
        except Exception as e:
            print(f"[DockerManager] No se pudo guardar active sessions: {e}")

    def _clear_symbol_history(self, symbol: str):
        """Elimina todos los registros de un símbolo en el diario de trades para iniciar limpio."""
        symbol_up = symbol.upper()
        if not self.journal_path.exists():
            return

        print(f"🧹 [DockerManager] Limpiando historial previo en diario para: {symbol_up}")
        try:
            temp_file = self.journal_path.with_suffix(".tmp_clean")
            with open(self.journal_path, "r", encoding="utf-8") as f_in, \
                 open(temp_file, "w", newline="", encoding="utf-8") as f_out:
                
                reader = csv.DictReader(f_in)
                writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
                writer.writeheader()
                
                rows_kept = 0
                for row in reader:
                    if row.get("symbol", "").upper() != symbol_up:
                        writer.writerow(row)
                        rows_kept += 1
            
            os.replace(temp_file, self.journal_path)
            print(f"✅ [DockerManager] Historial limpiado. Registros de otros símbolos conservados: {rows_kept}")
        except Exception as e:
            print(f"❌ [DockerManager] Error limpiando historial: {e}")

    def launch_bot(self, symbol: str, mode: str = "SIMULATED", start_date: str = None, end_date: str = None, freeze_learning: bool = False, stage: str = "TRAINING") -> tuple:
        container_name = f"hapi_worker_{mode.lower()}_{symbol.upper()}"

        # Guardar metadatos del lanzamiento INMEDIATAMENTE para el registro persistente
        self._launch_meta[symbol.upper()] = {
            "sim_start_date": start_date or "",
            "sim_end_date": end_date or "",
            "launched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stage": stage,               # 🏷️ Etapa del curriculum (TRAINING, VALIDATING, etc.)
            "freeze_learning": freeze_learning,  # 🔒 Si el aprendizaje está congelado
        }
        self._save_active_sessions()

        if not self.is_connected or self.client is None:
            self._connect_docker() # Reintento de último minuto
            if not self.is_connected:
                return False, "Docker daemon not reachable"

        # En el nuevo sistema de historial, NO borramos el historial anterior al relanzar.
        # Conservamos las simulaciones previas para ver la evolución.

        try:
            existing = self.client.containers.get(container_name)
            if existing.status == "running":
                return False, f"Bot for {symbol} is already running"
            else:
                existing.remove(force=True)
        except docker.errors.NotFound:
            pass

        # 🧹 LIMPIEZA PREVIA: Asegurar que el dashboard empiece de cero para este símbolo
        self._clear_symbol_history(symbol)

        try:
            command = "python simulation_mode/main_sim.py" if mode == "SIMULATED" else "python live_mode/main_live.py"

            container = self.client.containers.run(
                image="hapi-scalping-bot:latest",
                command=command,
                name=container_name,
                detach=True,
                remove=True,  # Auto-destrucción cuando el proceso Python termina
                environment={
                    "TARGET_SYMBOL": symbol,
                    "TRADING_MODE": mode,
                    "HAPI_IS_TEST_MODE": "true" if mode == "SIMULATED" else "false",
                    "HAPI_SIM_START_DATE": start_date or "",
                    "HAPI_SIM_END_DATE": end_date or "",
                    "ORCHESTRATOR_URL": "http://hapi_orchestrator:9000",
                    "TZ": "America/Bogota",
                    # 🔒 PVC: Congelar aprendizaje en fases de evaluación para evitar Data Leakage
                    "FREEZE_LEARNING": "true" if freeze_learning else "false",
                    # 🏷️ Etapa del curriculum para el histórico
                    "SIM_STAGE": stage or "TRAINING",
                },
                network="trading_bot_default",
                volumes={
                    '/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/data': {'bind': '/app/data', 'mode': 'rw'},
                    '/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/logs': {'bind': '/app/logs', 'mode': 'rw'},
                    '/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/.env': {'bind': '/app/.env', 'mode': 'ro'},
                    '/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/shared': {'bind': '/app/shared', 'mode': 'rw'},
                    '/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/simulation_mode': {'bind': '/app/simulation_mode', 'mode': 'rw'},
                    '/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/live_mode': {'bind': '/app/live_mode', 'mode': 'rw'}
                }
            )
            return True, f"Launched {symbol} (Container ID: {container.short_id})"
        except Exception as e:
            return False, str(e)

    def mark_completed(self, symbol: str, snapshot: Dict):
        """El Orquestador llama esto cuando el bot envía status=COMPLETED en su state update."""
        symbol_up = symbol.upper()
        meta = self._launch_meta.pop(symbol_up, {}) # Eliminar de activas al completar
        self._save_active_sessions()
        
        print(f"✅ [DockerManager] Marcando simulación como COMPLETADA para: {symbol_up}")
        
        import time
        run_id = f"{symbol_up}_{int(time.time())}"
        
        entry = {
            "run_id": run_id,
            "symbol": symbol_up,
            "sim_start_date": meta.get("sim_start_date", ""),
            "sim_end_date": meta.get("sim_end_date", ""),
            "launched_at": meta.get("launched_at", ""),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pnl": snapshot.get("total_sim_pnl", 0.0),
            "pnl_gross": snapshot.get("total_sim_gross_profit", 0.0),
            "trades": snapshot.get("total_sim_trades", 0),
            "win_rate": snapshot.get("win_rate", 0.0),
            "mode": snapshot.get("mode", "SIMULATED"),
            # --- Nuevos campos de Maestría ---
            "mastery_score": snapshot.get("mastery_score", 0.0),
            "mastery_status": snapshot.get("mastery_status", "APRENDIENDO"),
            "recommended_cash": snapshot.get("recommended_cash", 500.0),
            "actual_profit_factor": snapshot.get("actual_profit_factor", 0.0),
            "actual_max_drawdown": snapshot.get("actual_max_drawdown", 0.0),
            "mastery_checklist": snapshot.get("mastery_checklist", []),
            # --- Analyst Upgrade (Nuevos campos) ---
            "expectancy": snapshot.get("expectancy", 0.0),
            "efficiency": snapshot.get("efficiency", 0.0),
            "stability_score": snapshot.get("stability_score", 0.0),
            "equity_history": snapshot.get("equity_history", []),
            "trade_log": snapshot.get("trade_log", []),
            # 🏷️ PVC: Etapa del curriculum y modo de aprendizaje
            "stage": meta.get("stage", "TRAINING"),
            "freeze_learning": meta.get("freeze_learning", False),
        }
        
        self.completed_bots.insert(0, entry)
        if len(self.completed_bots) > 500:
            self.completed_bots = self.completed_bots[:500]
            
        self._save_completed()

    def dismiss_completed(self, run_id_or_symbol: str):
        """El usuario hace clic en Limpiar: elimina del registro de completados por run_id (o símbolo si es legacy)."""
        idx_to_remove = None
        for i, bot in enumerate(self.completed_bots):
            if bot.get("run_id") == run_id_or_symbol or bot.get("symbol") == run_id_or_symbol.upper():
                idx_to_remove = i
                break
                
        if idx_to_remove is not None:
            self.completed_bots.pop(idx_to_remove)
            self._save_completed()

    def kill_bot(self, symbol: str) -> tuple:
        if not self.is_connected:
            return False, "Docker connection error"

        for prefix in ["simulated", "live"]:
            container_name = f"hapi_worker_{prefix}_{symbol.upper()}"
            try:
                container = self.client.containers.get(container_name)
                container.stop(timeout=1)
                try:
                    container.remove(force=True)
                except Exception:
                    pass  # Si tiene auto_remove=True y ya lo está borrando docker, ignoramos el choque.
                return True, f"Killed worker for {symbol}"
            except docker.errors.NotFound:
                continue
            except Exception as e:
                return False, str(e)

    def kill_all_bots(self) -> List[str]:
        """Stops and removes all hapi_worker_* containers aggressively."""
        killed = []
        if not self.is_connected or not self.client:
            return killed
            
        try:
            # Filtramos todos los contenedores relacionados con el bot
            containers = self.client.containers.list(all=True, filters={"name": "hapi_worker_"})
            for c in containers:
                try:
                    # Usamos kill en lugar de stop para que sea instantáneo en el Wipe
                    c.kill()
                    c.remove(force=True)
                    killed.append(c.name)
                except Exception:
                    # Si ya estaba muerto o borrándose, lo ignoramos
                    try:
                        c.remove(force=True)
                    except: pass
        except Exception as e:
            print(f"⚠️ [DockerManager] Error en kill_all_bots masivo: {e}")

        return killed

    def list_active_bots(self) -> List[Dict]:
        """Retorna SOLO los contenedores actualmente vivos en Docker."""
        if not self.is_connected:
            return []

        try:
            containers = self.client.containers.list(filters={"name": "hapi_worker_"})
            results = []
            for c in containers:
                parts = c.name.split('_')
                if len(parts) >= 4:
                    sym = parts[3].upper()
                    mode = parts[2].upper()
                    
                    # Enriquecer con metadatos del lanzamiento (recuperados de disco si hubo reinicio)
                    meta = self._launch_meta.get(sym, {})
                    
                    results.append({
                        "id": c.short_id,
                        "name": c.name,
                        "symbol": sym,
                        "mode": mode,
                        "status": "running",
                        "sim_start_date": meta.get("sim_start_date", ""),
                        "sim_end_date": meta.get("sim_end_date", ""),
                        "launched_at": meta.get("launched_at", "—"),
                        "stage": meta.get("stage", "TRAINING"),
                        "freeze_learning": meta.get("freeze_learning", False)
                    })
            return results
        except Exception:
            return []

    def list_completed_bots(self) -> List[Dict]:
        """Retorna los bots que ya finalizaron su simulación en orden cronológico inverso."""
        return self.completed_bots

    # ─── System Health & Resource Governance ───────────────────────────────────

    def get_resource_health(self) -> Dict:
        """Calculates real-time CPU and RAM health metrics."""
        import psutil
        
        # 1. RAM
        mem = psutil.virtual_memory()
        total_ram_mb = mem.total // (1024 * 1024)
        used_ram_mb  = mem.used // (1024 * 1024)
        avail_ram_mb = mem.available // (1024 * 1024)
        ram_pct      = mem.percent

        # 2. CPU (quick sample pero más estable)
        # Un intervalo muy corto (0.1) causa que psutil reporte picos falsos de 70%+
        # debido a procesos de fondo de la Mac (Chrome, VirtualMachine base).
        cpu_pct = psutil.cpu_percent(interval=0.5)

        # 3. Decision Status
        if ram_pct < 65 and cpu_pct < 60:
            status = "ok"
            advice = "Sistema saludable"
        elif ram_pct < 85 and cpu_pct < 80:
            status = "warning"
            advice = "Recursos moderados"
        else:
            status = "critical"
            advice = "SISTEMA SATURADO"

        return {
            "status": status,
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
            "ram_used_mb": used_ram_mb,
            "ram_avail_mb": avail_ram_mb,
            "ram_total_mb": total_ram_mb,
            "advice": advice
        }

    def get_available_slots(self, base_limit: int = 4) -> int:
        """Determines how many new simulations can be launched SAFELY."""
        health = self.get_resource_health()
        
        if health["status"] == "critical":
            return 0
            
        # RAM Slots
        free_ram_for_sims = max(0, health["ram_avail_mb"] - self._MIN_FREE_RAM_MB)
        ram_slots = int(free_ram_for_sims // self._RAM_PER_SIM_MB)
        
        # CPU Slots
        free_cpu = max(0, 95 - health["cpu_pct"]) # Leave 5% breathing room
        cpu_slots = int(free_cpu // self._CPU_PER_SIM_PCT)
        
        max_safe = min(ram_slots, cpu_slots)
        
        # En TURBO, dejamos que el hardware sea el límite real ignorando el base_limit si es saludable
        # (Pero mantenemos un tope máximo de 15 para evitar saturación de E/S de disco)
        effective_limit = 15 if base_limit > 10 else base_limit
        
        return min(effective_limit, max_safe)
