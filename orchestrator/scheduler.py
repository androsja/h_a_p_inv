import asyncio
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

SCHEDULE_LOG = Path("/app/data/schedules.json")

@dataclass
class Job:
    id: str           # UUID
    symbol: str
    mode: str         # SIMULATED / LIVE
    frequency: str    # HOURLY, DAILY, WEEKLY
    next_run: str     # ISO Timestamp
    last_run: Optional[str] = None
    status: str = "SCHEDULED" # SCHEDULED, RUNNING, ERROR

class ScheduleManager:
    def __init__(self, docker_mgr):
        self.docker_mgr = docker_mgr
        self.jobs: Dict[str, Job] = self._load_jobs()
        self._running = False
        self._loop_task = None

    def _load_jobs(self) -> Dict[str, Job]:
        try:
            if SCHEDULE_LOG.exists():
                data = json.loads(SCHEDULE_LOG.read_text(encoding="utf-8"))
                return {k: Job(**v) for k, v in data.items()}
        except Exception as e:
            print(f"⚠️ [ScheduleManager] Error cargando tareas: {e}")
        return {}

    def _save_jobs(self):
        try:
            SCHEDULE_LOG.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self.jobs.items()}
            SCHEDULE_LOG.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ [ScheduleManager] Error guardando tareas: {e}")

    def add_job(self, symbol: str, mode: str, frequency: str):
        job_id = str(uuid.uuid4())[:8]
        
        # Calcular primer ejecución
        next_dt = datetime.now()
        if frequency == "HOURLY": next_dt += timedelta(hours=1)
        elif frequency == "DAILY": next_dt += timedelta(days=1)
        elif frequency == "WEEKLY": next_dt += timedelta(weeks=1)
        
        new_job = Job(
            id=job_id,
            symbol=symbol.upper(),
            mode=mode.upper(),
            frequency=frequency.upper(),
            next_run=next_dt.isoformat()
        )
        self.jobs[job_id] = new_job
        self._save_jobs()
        return job_id

    def remove_job(self, job_id: str):
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save_jobs()
            return True
        return False

    def list_jobs(self) -> List[Job]:
        return list(self.jobs.values())

    async def start(self):
        if not self._running:
            self._running = True
            self._loop_task = asyncio.create_task(self._main_loop())
            print("🚀 [ScheduleManager] Motor de automatización iniciado.")

    async def stop(self):
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()

    async def _main_loop(self):
        """Ciclo de fondo que revisa el reloj cada 30 segundos."""
        while self._running:
            now = datetime.now()
            for job_id, job in self.jobs.items():
                try:
                    next_run_dt = datetime.fromisoformat(job.next_run)
                    if now >= next_run_dt and job.status != "RUNNING":
                        # ¡Es hora de ejecutar!
                        await self._execute_job(job)
                except Exception as e:
                    print(f"❌ [ScheduleManager] Error en job {job_id}: {e}")
            
            await asyncio.sleep(30) # Revisar cada medio minuto

    async def _execute_job(self, job: Job):
        print(f"📅 [ScheduleManager] Ejecutando tarea programada: {job.symbol} ({job.frequency})")
        job.status = "RUNNING"
        job.last_run = datetime.now().isoformat()
        
        # Actualizar próxima ejecución
        now = datetime.now()
        if job.frequency == "HOURLY": next_dt = now + timedelta(hours=1)
        elif job.frequency == "DAILY": next_dt = now + timedelta(days=1)
        elif job.frequency == "WEEKLY": next_dt = now + timedelta(weeks=1)
        else: next_dt = now + timedelta(days=1) # Default
        
        job.next_run = next_dt.isoformat()
        self._save_jobs()

        # Lanzar el bot vía DockerManager
        success, msg = self.docker_mgr.launch_bot(job.symbol, job.mode)
        if success:
            print(f"✅ [ScheduleManager] Bot lanzado con éxito para {job.symbol}")
            job.status = "RUNNING"
        else:
            print(f"❌ [ScheduleManager] Fallo al lanzar bot para {job.symbol}: {msg}")
            # Si ya está corriendo, no lo marcamos como ERROR, lo dejamos como RUNNING o SCHEDULED
            if "already running" in msg.lower():
                job.status = "RUNNING"
            else:
                job.status = "ERROR"
        
        self._save_jobs()

        # Volvemos a SCHEDULED tras lanzar (el orquestador se encargará de marcarlo como completado en el hub al terminar)
        # Nota: Si falló seriamente (ERROR), se queda en ERROR para que el usuario lo vea.
        if job.status == "RUNNING":
            job.status = "SCHEDULED"
        self._save_jobs()

    def run_job_now(self, job_id: str) -> bool:
        """Fuerza la ejecución de una tarea inmediatamente."""
        if job_id not in self.jobs:
            return False
            
        job = self.jobs[job_id]
        print(f"⚡ [ScheduleManager] Forzando ejecución inmediata de {job.symbol}")
        
        # Ejecutar de forma asíncrona sin bloquear la petición web
        asyncio.create_task(self._execute_job(job))
        return True
