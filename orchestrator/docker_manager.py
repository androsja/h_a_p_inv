import os
import docker
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import csv
from shared import config

COMPLETED_LOG = Path("/app/data/completed_simulations.json")
ACTIVE_SESSIONS_LOG = Path("/app/data/active_sessions.json")

class DockerManager:
    def __init__(self):
        self.is_connected = False
        self.client = None
        self._connect_docker()
        self.journal_path = config.TRADE_JOURNAL_FILE

        # Registro persistente de bots completados (cargado desde disco)
        # Estructura: List of dicts, cronológicamente ordenado
        self.completed_bots: List[Dict] = self._load_completed()

        # Metadatos de lanzamientos activos (Persistido en disco para sobrevivir reinicios)
        # Estructura: {symbol: {"sim_start_date": str, "launched_at": str}}
        self._launch_meta: Dict[str, Dict] = self._load_active_sessions()

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
        """Carga los metadatos de sesiones activas desde disco."""
        try:
            if ACTIVE_SESSIONS_LOG.exists():
                return json.loads(ACTIVE_SESSIONS_LOG.read_text(encoding="utf-8"))
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

    def launch_bot(self, symbol: str, mode: str = "SIMULATED", start_date: str = None) -> tuple:
        container_name = f"hapi_worker_{mode.lower()}_{symbol.upper()}"

        # Guardar metadatos del lanzamiento INMEDIATAMENTE para el registro persistente
        self._launch_meta[symbol.upper()] = {
            "sim_start_date": start_date or "",
            "launched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
                    "ORCHESTRATOR_URL": "http://hapi_orchestrator:9000",
                    "TZ": "America/Bogota"
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
            "launched_at": meta.get("launched_at", ""),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pnl": snapshot.get("total_sim_pnl", 0.0),
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
            "equity_history": snapshot.get("equity_history", [])
        }
        
        self.completed_bots.insert(0, entry)
        if len(self.completed_bots) > 50:
            self.completed_bots = self.completed_bots[:50]
            
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

        return False, f"No running container found for {symbol}"

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
                        "sim_start_date": meta.get("sim_start_date", "—"),
                        "launched_at": meta.get("launched_at", "—")
                    })
            return results
        except Exception:
            return []

    def list_completed_bots(self) -> List[Dict]:
        """Retorna los bots que ya finalizaron su simulación en orden cronológico inverso."""
        return self.completed_bots
