import asyncio
import json
import time
try:
    import psutil
except ImportError:
    psutil = None
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

from shared import config
from orchestrator.bank import BankManager
from orchestrator.docker_manager import DockerManager
from orchestrator.state_manager import StateManager
from orchestrator.scheduler import ScheduleManager
from orchestrator.auto_trainer import AutoTrainerManager
from orchestrator.mastery_manager import MasteryManager
from shared.utils.neural_filter import get_neural_filter
from orchestrator.ai_analyst import hapi_ai
from orchestrator.queue_manager import QueueManager

app = FastAPI(title="Hapi Orchestrator")

# Habilitar CORS para que el dashboard (Puerto 8080) pueda consumir la API y el WS (Puerto 9000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the orchestrator dashboard
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
if not (STATIC_DIR / "index.html").exists():
    (STATIC_DIR / "index.html").write_text("<html><body>Orchestrator Panel (Building...)</body></html>")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Managers
bank = BankManager(initial_cash=10000.0)
docker_mgr = DockerManager()
state_mgr = StateManager()
mastery_mgr = MasteryManager()
queue_mgr = QueueManager()

# Inyección de dependencias
bank.mastery_mgr = mastery_mgr
scheduler_mgr = ScheduleManager(docker_mgr)
auto_trainer_mgr = AutoTrainerManager(docker_mgr)
auto_trainer_mgr.mastery_mgr = mastery_mgr
auto_trainer_mgr.queue_mgr = queue_mgr

# --- Bank API ---
class TransactionRequest(BaseModel):
    symbol: str
    amount: float
    type: str # 'buy' or 'sell' (or 'profit')

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/bank/balance")
async def get_balance():
    return {
        "status": "success", 
        "available_cash": bank.available_cash,
        "initial_cash": bank.initial_cash,
        "total_pnl": bank.available_cash - bank.initial_cash
    }

@app.post("/api/bank/transaction")
async def process_transaction(req: TransactionRequest):
    """
    Called by bots to request funds for a trade, or report a closed position (cash return).
    """
    success, msg = bank.process_transaction(req.symbol, req.amount, req.type)
    if not success:
        return JSONResponse(status_code=400, content={"status": "error", "message": msg})
    return {"status": "success", "message": msg, "available_cash": bank.available_cash}

# --- Docker API ---
class LaunchRequest(BaseModel):
    symbol: str
    mode: str = "SIMULATED"
    start_date: str = None  # Formato YYYY-MM-DD
    end_date: str = None    # Formato YYYY-MM-DD
    freeze_learning: bool = False
    stage: str = "TRAINING"

@app.post("/api/docker/launch")
async def launch_container(req: LaunchRequest):
    # 🧹 Limpiar caché de estado antes de lanzar para que el UI se vea limpio inmediatamente
    await state_mgr.clear_symbol(req.symbol)
    
    # 📏 Verificar disponibilidad de recursos
    # Si no hay slots, lo mandamos a la cola
    available = docker_mgr.get_available_slots()
    if available <= 0:
        added = queue_mgr.add_to_queue(req.model_dump())
        if added:
            return {"status": "queued", "message": f"{req.symbol} añadido a la cola por falta de recursos."}
        else:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"{req.symbol} ya está en cola o en ejecución."})

    success, msg = docker_mgr.launch_bot(
        req.symbol, 
        req.mode, 
        req.start_date, 
        req.end_date, 
        freeze_learning=req.freeze_learning,
        stage=req.stage
    )
    if not success:
         return JSONResponse(status_code=500, content={"status": "error", "message": msg})
    return {"status": "success", "message": msg}

# --- Queue API ---
@app.get("/api/queue/list")
async def get_queue():
    return {"status": "success", "queue": queue_mgr.get_queue()}

@app.delete("/api/queue/remove/{symbol}")
async def remove_from_queue(symbol: str):
    success = queue_mgr.remove_from_queue(symbol)
    if success:
        return {"status": "success", "message": f"{symbol} eliminado de la cola."}
    return JSONResponse(status_code=404, content={"status": "error", "message": "No encontrado en cola."})

@app.post("/api/docker/kill/{symbol}")
async def kill_container(symbol: str):
    success, msg = docker_mgr.kill_bot(symbol)
    return {"status": "success", "message": msg}

# --- AI Central Brain API ---
class PredictRequest(BaseModel):
    features: List[float]

@app.post("/api/ai/predict")
async def ai_predict(req: PredictRequest):
    nf = get_neural_filter("GLOBAL")
    proba, reason = nf.predict(req.features)
    return {"proba": proba, "reason": reason}

# --- Mastery Hub API ---
@app.get("/api/mastery/rankings")
async def get_mastery_rankings():
    res = mastery_mgr.get_rankings()
    return {
        "status": "success",
        "rankings": res["rankings"],
        "summary": res["summary"],
        "recommendations": mastery_mgr.get_elite_recommendations(limit=5)
    }

@app.get("/api/mastery/exposure")
async def get_exposure():
    return {
        "status": "success",
        "exposure": bank.get_sector_exposure()
    }

class TrainRequest(BaseModel):
    features: List[float]
    won: bool

from fastapi import BackgroundTasks

@app.post("/api/ai/train")
async def ai_train(req: TrainRequest, background_tasks: BackgroundTasks):
    nf = get_neural_filter("GLOBAL")
    # Mover el entrenamiento y el guardado a segundo plano para no bloquear al worker
    background_tasks.add_task(nf.fit, req.features, req.won)
    return {"status": "success"}

@app.get("/api/docker/status")
async def get_docker_status():
    containers = docker_mgr.list_active_bots()
    completed = docker_mgr.list_completed_bots()
    
    # 📡 FALLBACK POR LATIDO (HEARTBEAT)
    # Si Docker no responde o la lista está vacía, buscamos quién está "vivo" en memoria
    active_symbols = set(c.get("symbol", "").upper() for c in containers)
    
    for sym, live_data in state_mgr.global_state.items():
        if sym == "_main": continue
        if sym not in active_symbols:
            # 🕒 Verificar frescura del latido (Filtro Anti-Fantasmas)
            # Solo permitimos fallback si el bot ha reportado en los últimos 60 segundos
            last_report = state_mgr.last_updates.get(sym, 0)
            is_recent = (time.time() - last_report) < 60
            
            if is_recent:
                containers.append({
                    "symbol": sym,
                    "status": "running",
                    "mode": live_data.get("mode", "SIMULATED"),
                    "stage": live_data.get("stage") or live_data.get("stage", "UNKNOWN"), # Prioridad al latido real
                    "launched_at": live_data.get("launched_at", "Heartbeat Detected"),
                    "is_fallback": True 
                })
                active_symbols.add(sym)
            else:
                # Si el bot no es reciente, se ignora (se considera muerto tras el reinicio/error)
                pass
    # ─── ENRIQUECER CONTENEDORES ACTIVOS CON MÉTRICAS EN TIEMPO REAL ───
    # Buscamos en el global_state el símbolo correspondiente a cada contenedor.
    for c in containers:
        sym = c.get("symbol", "").upper()
        # El StateManager guarda snapshots enviadas por los runners.
        live_data = state_mgr.global_state.get(sym, {})
        if live_data:
            c["pnl_net"]   = live_data.get("total_sim_pnl", 0.0)
            c["pnl_gross"] = live_data.get("total_sim_gross_pnl", 0.0)
            c["trades"]    = live_data.get("total_sim_trades", 0)
            
            # --- NUEVAS MÉTRICAS DE SESIÓN ---
            c["session_trades"] = live_data.get("total_trades", 0)
            s_gp = live_data.get("gross_profit", 0.0)
            s_gl = abs(live_data.get("gross_loss", 0.0))
            s_fees = live_data.get("total_fees", 0.0)
            s_slip = live_data.get("total_slippage", 0.0)
            c["session_pnl_net"] = round((s_gp - s_gl) - s_fees - s_slip, 2)
            
            c["win_rate"]  = live_data.get("win_rate", 0.0)
            c["progress"]  = live_data.get("sim_progress_pct", 0.0)
            c["score"]     = live_data.get("mastery_score", 0.0)
            
            # Calcular ETA si el progreso es válido y tenemos la fecha de lanzamiento
            if c.get("launched_at") and c.get("launched_at") != "—" and c["progress"] > 0:
                try:
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M:%S"
                    start = datetime.strptime(c["launched_at"], fmt)
                    elapsed = (datetime.now() - start).total_seconds()
                    total_estimated = elapsed / (c["progress"] / 100.0)
                    remaining = total_estimated - elapsed
                    if remaining > 0:
                        rm_m = int(remaining // 60)
                        rm_s = int(remaining % 60)
                        c["eta_str"] = f"{rm_m}m {rm_s}s"
                except: pass

    # Agregar métricas de la IA Central
    try:
        nf = get_neural_filter("GLOBAL")
        ai_stats = nf.get_stats()
    except Exception as e:
        print(f"⚠️ [API] Error al obtener stats de IA: {e}")
        ai_stats = {"ai_mode": "MLP", "is_frozen": False, "total_samples": 0}

    return {
        "status": "success", 
        "containers": containers, 
        "completed": completed,
        "ai": ai_stats
    }

@app.post("/api/docker/launch")
async def launch_bot(req: Dict):
    symbol = req.get("symbol")
    mode = req.get("mode", "SIMULATED")
    start_date = req.get("start_date")
    end_date = req.get("end_date")
    freeze = req.get("freeze_learning", False)
    stage = req.get("stage", "TRAINING")

    if not symbol:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Symbol is required"})
    
    success, msg = docker_mgr.launch_bot(symbol, mode, start_date, end_date, freeze_learning=freeze, stage=stage)
    return {"status": "success" if success else "error", "message": msg}

@app.post("/api/docker/kill/{symbol}")
async def kill_bot(symbol: str):
    success, msg = docker_mgr.kill_bot(symbol)
    if success:
        # 🧹 Limpiar rastro de memoria al instante para evitar "bots fantasma"
        await state_mgr.clear_symbol(symbol)
    return {"status": "success" if success else "error", "message": msg}

@app.get("/api/docker/logs/{symbol}")
async def get_bot_logs(symbol: str):
    logs = docker_mgr.get_container_logs(symbol)
    return {"status": "success", "symbol": symbol, "logs": logs}

@app.post("/api/docker/dismiss/{symbol}")
async def dismiss_completed(symbol: str):
    """El usuario limpia el registro de un bot completado desde la consola del Orquestador."""
    docker_mgr.dismiss_completed(symbol.upper())
    return {"status": "success", "message": f"{symbol} removed from completed registry"}

@app.get("/api/symbols")
async def get_symbols():
    assets_path = Path("/app/assets.json")
    if assets_path.exists():
        try:
            data = json.loads(assets_path.read_text(encoding="utf-8"))
            return {"status": "success", "assets": data.get("assets", [])}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "assets.json not found"}

# --- Scheduler API ---
class ScheduleJobRequest(BaseModel):
    symbol: str
    mode: str = "SIMULATED"
    frequency: str = "DAILY"

@app.get("/api/scheduler/jobs")
async def get_scheduler_jobs():
    return {"status": "success", "jobs": scheduler_mgr.list_jobs()}

@app.post("/api/scheduler/jobs")
async def add_scheduler_job(req: ScheduleJobRequest):
    job_id = scheduler_mgr.add_job(req.symbol, req.mode, req.frequency)
    return {"status": "success", "job_id": job_id}

@app.delete("/api/scheduler/jobs/{job_id}")
async def delete_scheduler_job(job_id: str):
    success = scheduler_mgr.remove_job(job_id)
    if not success:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Job not found"})
    return {"status": "success", "message": f"Job {job_id} deleted"}

@app.post("/api/scheduler/jobs/{job_id}/run")
async def run_scheduler_job_now(job_id: str):
    success = scheduler_mgr.run_job_now(job_id)
    if not success:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Job not found"})
    return {"status": "success", "message": f"Job {job_id} triggered immediately"}

# --- Auto-Trainer API ---
@app.get("/api/auto-trainer/status")
async def get_auto_trainer_status():
    return {"status": "success", "data": auto_trainer_mgr.get_status()}

class AutoTrainerSettingsRequest(BaseModel):
    max_bots: int

@app.post("/api/auto-trainer/settings")
async def update_auto_settings(req: Dict):
    max_bots = req.get("max_bots", 4)
    auto_trainer_mgr.update_settings(max_bots=max_bots)
    return {"status": "success"}

@app.post("/api/auto-trainer/profile")
async def update_auto_profile(req: Dict):
    profile = req.get("profile", "turbo")
    auto_trainer_mgr.update_performance_profile(profile)
    return {"status": "success"}

@app.post("/api/auto-trainer/toggle")
async def toggle_auto_trainer():
    is_running = auto_trainer_mgr.toggle()
    return {"status": "success", "is_running": is_running}

@app.post("/api/auto-trainer/force-sim")
async def toggle_force_sim(req: Dict):
    force_sim = req.get("force_sim", False)
    auto_trainer_mgr.update_safety_mode(force_sim)
    return {"status": "success"}

# --- AI Analyst API ---
@app.get("/api/ai/analyze/global")
async def analyze_global_operation():
    """
    Realiza una auditoría completa de todo el sistema (Banco, Bots, IA).
    """
    # 1. Datos del banco
    bank_data = {
        "available_cash": bank.available_cash,
        "initial_cash": bank.initial_cash,
        "total_pnl": bank.available_cash - bank.initial_cash
    }
    
    # 2. Bots activos y métricas
    status = await get_docker_status()
    containers = status.get("containers", [])
    completed = status.get("completed", [])
    
    # 3. IA Central
    nf = get_neural_filter("GLOBAL")
    ai_stats = nf.get_stats()
    
    # 4. Auditoría
    audit_text = await hapi_ai.analyze_global(containers, completed, bank_data, ai_stats)
    
    if audit_text.startswith("ERROR"):
        status_code = 429 if "429" in audit_text else 500
        return JSONResponse(status_code=status_code, content={"status": "error", "message": audit_text})

    return {
        "status": "success",
        "audit": audit_text
    }

@app.get("/api/ai/analyze/{symbol}")
async def analyze_bot_performance(symbol: str):
    sym = symbol.upper()
    
    # 1. Obtener datos de estado vivo
    live_data = state_mgr.global_state.get(sym, {})
    if not live_data:
        # Intentar obtener de contenedores si no hay estado vivo aún
        containers = docker_mgr.list_active_bots()
        bot = next((c for c in containers if c["symbol"] == sym), None)
        if not bot:
             return JSONResponse(status_code=404, content={"status": "error", "message": f"No se encontró proceso activo para {sym}"})
        live_data = bot
    
    # 2. Obtener métricas de maestría
    mastery_data = mastery_mgr.get_symbol_status(sym)
    
    # 3. Solicitar auditoría a Gemini
    audit_text = await hapi_ai.analyze_simulation(sym, live_data, mastery_data)
    
    if audit_text.startswith("ERROR"):
        status_code = 429 if "429" in audit_text else 500
        return JSONResponse(status_code=status_code, content={"status": "error", "message": audit_text})

    return {
        "status": "success",
        "symbol": sym,
        "audit": audit_text
    }

# --- State Sync (Dashboard / UI) API ---
class StateUpdateRequest(BaseModel):
    symbol: str
    data: Dict[str, Any]

@app.post("/api/state/update")
async def update_sim_state(req: StateUpdateRequest):
    # Usar una tarea separada para la actualización para liberar el hilo rápido
    asyncio.create_task(state_mgr.update_state(req.symbol, req.data))
    
    # Detectar si el bot acabó.
    flat_status = str(req.data.get("status", "")).lower()
    if flat_status == "completed":
        # Consolidar: Usar el estado acumulado en memoria en lugar de solo el snapshot parcial recibido
        full_snapshot = state_mgr.global_state.get(req.symbol, req.data)
        docker_mgr.mark_completed(req.symbol, full_snapshot)
    else:
        # Búsqueda en payloads multi-key (legacy)
        for _key, obj in req.data.items():
            if isinstance(obj, dict) and str(obj.get("status", "")).lower() == "completed":
                full_snapshot = state_mgr.global_state.get(req.symbol, obj)
                docker_mgr.mark_completed(req.symbol, full_snapshot)
                break
    
    return {"status": "success"}

@app.get("/api/state/current")
async def get_current_state():
    # Retorna la snapshot inmediata de todo el estado en memoria para el front-end
    return state_mgr.global_state

@app.post("/api/state/clear")
async def clear_sim_state():
    await state_mgr.clear()
    return {"status": "success"}

@app.websocket("/api/state/stream")
async def websocket_state_stream(websocket: WebSocket):
    await state_mgr.connect(websocket)
    try:
        while True:
            # Mantener viva la conexión
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await state_mgr.disconnect(websocket)
    except Exception:
        await state_mgr.disconnect(websocket)


# ─── System Health API ─────────────────────────────────────────────────────
# RAM/CPU limits: each simulation container uses ~400MB RAM and ~10% CPU on average
_RAM_PER_SIM_MB  = 400
_CPU_PER_SIM_PCT = 10
_MIN_FREE_RAM_MB = 512  # always keep 512MB free as buffer

@app.get("/api/system/health")
async def get_system_health():
    """Returns real-time CPU, RAM and Docker container stats + recommended sim capacity."""
    health = docker_mgr.get_resource_health()
    active_containers = docker_mgr.list_active_bots()
    running_count = len(active_containers)
    max_new = docker_mgr.get_available_slots(base_limit=8)

    return {
        "status": health["status"],
        "cpu_pct": health["cpu_pct"],
        "ram_pct": health["ram_pct"],
        "ram_used_mb": health["ram_used_mb"],
        "ram_avail_mb": health["ram_avail_mb"],
        "ram_total_mb": health["ram_total_mb"],
        "running_sims": running_count,
        "max_new_sims": max_new,
        "advice": health["advice"],
    }

@app.post("/api/system/wipe")
async def system_wipe(req: Dict = None):
    """Borra el estado actual, con opción de proteger el historial (PVC)."""
    # Por defecto protegemos la historia
    keep_history = req.get("keep_history", True) if req else True
    
    from orchestrator.nuclear_wipe import nuclear_wipe_execution
    from shared.utils.neural_filter import reset_neural_filter
    import asyncio
    
    def perform_cleanup_pvc():
        try:
            # 1. Ejecutar limpieza física unificada (Docker + Files)
            nuclear_wipe_execution(keep_history=keep_history)
            
            # 2. Reiniciar estados en memoria que siempre se limpian
            bank.reset()
            docker_mgr.clear_memory()
            
            # 3. Solo resetear inteligencia acumulada si el usuario lo pidió explícitamente
            if not keep_history:
                mastery_mgr.reset()
                reset_neural_filter()
                
            return True
        except Exception as e:
            print(f"❌ Error en Limpieza PVC: {e}")
            return False

    # Ejecutar en segundo plano para no bloquear la respuesta
    asyncio.create_task(asyncio.to_thread(perform_cleanup_pvc))
    await state_mgr.clear()
    
    msg = "Limpieza completada (Historial Protegido)" if keep_history else "SISTEMA PURGADO (Borrado Total)"
    log.info(f"💣 {msg}")
    return {"status": "success", "message": msg, "keep_history": keep_history}

# UI Route

@app.get("/")
async def serve_ui():
    content = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=content)

@app.on_event("startup")
async def startup():
    print("🚀 Orchestrator Master Node started on port 9000.")
    await scheduler_mgr.start()

@app.on_event("shutdown")
async def shutdown():
    await scheduler_mgr.stop()
    print("👋 Orchestrator shutting down.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

