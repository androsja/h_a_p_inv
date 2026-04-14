import asyncio
import json
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
from shared.utils.neural_filter import get_neural_filter

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
scheduler_mgr = ScheduleManager(docker_mgr)

# --- Bank API ---
class TransactionRequest(BaseModel):
    symbol: str
    amount: float
    type: str # 'buy' or 'sell' (or 'profit')

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

@app.post("/api/docker/launch")
async def launch_container(req: LaunchRequest):
    # 🧹 Limpiar caché de estado antes de lanzar para que el UI se vea limpio inmediatamente
    await state_mgr.clear_symbol(req.symbol)
    
    success, msg = docker_mgr.launch_bot(req.symbol, req.mode, req.start_date)
    if not success:
         return JSONResponse(status_code=500, content={"status": "error", "message": msg})
    return {"status": "success", "message": msg}

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
    return {"status": "success", "proba": proba, "reason": reason}

class TrainRequest(BaseModel):
    features: List[float]
    won: bool

@app.post("/api/ai/train")
async def ai_train(req: TrainRequest):
    nf = get_neural_filter("GLOBAL")
    nf.fit(req.features, req.won)
    return {"status": "success"}

@app.get("/api/docker/status")
async def get_docker_status():
    containers = docker_mgr.list_active_bots()
    completed = docker_mgr.list_completed_bots()
    
    # Agregar métricas de la IA Central
    nf = get_neural_filter("GLOBAL")
    ai_stats = nf.get_stats()
    
    return {
        "status": "success", 
        "containers": containers, 
        "completed": completed,
        "ai": ai_stats
    }

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

# --- State Sync (Dashboard / UI) API ---
class StateUpdateRequest(BaseModel):
    symbol: str
    data: Dict[str, Any]

@app.post("/api/state/update")
async def update_sim_state(req: StateUpdateRequest):
    await state_mgr.update_state(req.symbol, req.data)
    
    # Detectar si el bot acabó.
    # Caso 1: payload plano (el runner envía {status: "completed", total_sim_pnl: ...})
    flat_status = str(req.data.get("status", "")).lower()
    if flat_status == "completed":
        docker_mgr.mark_completed(req.symbol, req.data)
    else:
        # Caso 2: payload anidado {symbol_key: {status: "COMPLETED", ...}}
        for _key, obj in req.data.items():
            if isinstance(obj, dict) and str(obj.get("status", "")).lower() == "completed":
                docker_mgr.mark_completed(req.symbol, obj)
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

