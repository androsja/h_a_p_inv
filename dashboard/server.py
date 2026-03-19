"""
dashboard/server.py ─ Servidor FastAPI + WebSocket para el dashboard en tiempo real.

Responsabilidades de ESTE archivo:
  - Montar archivos estáticos
  - Gestionar conexiones WebSocket (ConnectionManager)
  - Broadcaster de estado en background (lee state files y empuja via WS)
  - Endpoints de páginas HTML (/ y /live)
  - Endpoint /api/state
  - Registrar los APIRouters por dominio (simulación, activos, banco, snapshots)

Todo lo demás está en dashboard/routers/.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from shared import config
from shared.config import (
    STATE_FILE, LOG_FILE, COMMAND_FILE,
    STATE_FILE_SIM, STATE_FILE_LIVE
)

# ─── Routers por dominio ──────────────────────────────────────────────────────
from dashboard.routers.simulation_router import router as sim_router
from dashboard.routers.assets_router import router as assets_router
from dashboard.routers.bank_router import router as bank_router
from dashboard.routers.snapshots_router import router as snapshots_router

SIM_HTML_FILE  = Path(__file__).parent / "static" / "simulation" / "index.html"
LIVE_HTML_FILE = Path(__file__).parent / "static" / "live" / "index.html"

app = FastAPI(title="Hapi Bot Dashboard")

# ─── Archivos estáticos ───────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ─── Registrar Routers ────────────────────────────────────────────────────────
app.include_router(sim_router)
app.include_router(assets_router)
app.include_router(bank_router)
app.include_router(snapshots_router)


# ─── Manager de conexiones WebSocket ─────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {"sim": [], "live": []}

    async def connect(self, ws: WebSocket, mode: str):
        await ws.accept()
        if mode not in self.active:
            self.active[mode] = []
        self.active[mode].append(ws)

    def disconnect(self, ws: WebSocket):
        for mode in self.active:
            if ws in self.active[mode]:
                self.active[mode].remove(ws)
                break

    async def broadcast(self, data, mode: str):
        if mode not in self.active:
            return
        dead = []
        for ws in self.active[mode]:
            try:
                if isinstance(data, str):
                    await ws.send_text(data)
                else:
                    await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── Lectura del estado del bot ───────────────────────────────────────────────
def read_state(mode: str = "sim") -> dict:
    state = {}
    target_file = config.STATE_FILE_SIM if mode == "sim" else config.STATE_FILE_LIVE
    try:
        if target_file.exists():
            with open(target_file) as f:
                state = json.load(f)
    except Exception:
        state = {"status": "starting", "timestamp": datetime.utcnow().isoformat()}

    if COMMAND_FILE.exists():
        try:
            with open(COMMAND_FILE) as f:
                cmds = json.load(f)
                state["force_symbols"] = cmds.get("force_symbols", [])
        except: pass

    return state


def read_last_logs(n: int = 1000) -> list[str]:
    """Lee las últimas n líneas del log principal. Filtra líneas HTML de error."""
    HTML_NOISE = ("<html>", "<head>", "<body>", "</html>", "</head>", "</body>",
                  "<center>", "</center>", "<hr>", "<h1>", "nginx", "<!DOCTYPE")

    def _is_valid_log_line(line: str) -> bool:
        s = line.strip()
        if not s: return False
        sl = s.lower()
        return not any(token.lower() in sl for token in HTML_NOISE)

    all_lines = []
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                all_lines.extend(f.readlines())

        log_dir = LOG_FILE.parent
        for symbol_log in log_dir.glob("historial_*.log"):
            try:
                with open(symbol_log, "r", encoding="utf-8", errors="replace") as f:
                    all_lines.extend(f.readlines())
            except: pass

        if not all_lines:
            return ["(Esperando logs del bot...)"]

        all_lines.sort()
        return [l.strip() for l in all_lines[-n * 3:] if _is_valid_log_line(l)][-n:]

    except Exception as e:
        return [f"Error leyendo logs: {str(e)}"]

# ─── Broadcaster en background ────────────────────────────────────────────────
async def state_broadcaster():
    """Lee periódicamente los state files y los empuja a los WebSocket clientes."""
    last_sim_mtime = 0
    last_live_mtime = 0

    while True:
        try:
            if config.STATE_FILE_SIM.exists():
                mtime = config.STATE_FILE_SIM.stat().st_mtime
                if mtime > last_sim_mtime:
                    last_sim_mtime = mtime
                    try:
                        content = config.STATE_FILE_SIM.read_text(encoding="utf-8")
                        data = json.loads(content)
                        data["logs"] = read_last_logs()
                        await manager.broadcast(data, mode="sim")
                    except Exception as e_json:
                        print(f"Error parse sim state: {e_json}")
                        await manager.broadcast(content, mode="sim")
            elif last_sim_mtime > 0:
                last_sim_mtime = 0
                await manager.broadcast(json.dumps({}), mode="sim")

            if config.STATE_FILE_LIVE.exists():
                mtime = config.STATE_FILE_LIVE.stat().st_mtime
                if mtime > last_live_mtime:
                    last_live_mtime = mtime
                    try:
                        content = config.STATE_FILE_LIVE.read_text(encoding="utf-8")
                        data = json.loads(content)
                        data["logs"] = read_last_logs()
                        await manager.broadcast(data, mode="live")
                    except Exception as e_json:
                        print(f"Error parse live state: {e_json}")
                        await manager.broadcast(content, mode="live")
            elif last_live_mtime > 0:
                last_live_mtime = 0
                await manager.broadcast(json.dumps({}), mode="live")

        except Exception as e:
            print(f"Error en broadcaster: {e}")

        await asyncio.sleep(0.5)


# ─── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    asyncio.create_task(state_broadcaster())


# ─── Endpoints de páginas HTML ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def simulation_index():
    if not SIM_HTML_FILE.exists():
        return HTMLResponse("Simulation UI missing", status_code=404)
    return HTMLResponse(content=SIM_HTML_FILE.read_text(encoding="utf-8"))


@app.get("/live", response_class=HTMLResponse)
async def live_index():
    if not LIVE_HTML_FILE.exists():
        return HTMLResponse("Live UI missing", status_code=404)
    return HTMLResponse(content=LIVE_HTML_FILE.read_text(encoding="utf-8"))


# ─── WebSocket endpoints ──────────────────────────────────────────────────────
@app.websocket("/ws/sim")
async def websocket_sim(websocket: WebSocket):
    await manager.connect(websocket, mode="sim")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] WS_SIM: Cliente conectado. Intentando inyección inicial...")
    
    # Enviar estado actual INMEDIATAMENTE al conectar para evitar pantalla "Cargando..."
    try:
        if config.STATE_FILE_SIM.exists():
            content = config.STATE_FILE_SIM.read_text(encoding="utf-8")
            data = json.loads(content)
            data["logs"] = read_last_logs()
            await websocket.send_json(data)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WS_SIM: Inyección inicial enviada ({len(content)} bytes)")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WS_SIM: Archivo de estado NO hallado en {config.STATE_FILE_SIM}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] WS_SIM: Error en inyección inicial: {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] WS_SIM: Cliente desconectado")
        manager.disconnect(websocket)


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await manager.connect(websocket, mode="live")
    
    # Enviar estado actual INMEDIATAMENTE al conectar
    try:
        if config.STATE_FILE_LIVE.exists():
            content = config.STATE_FILE_LIVE.read_text(encoding="utf-8")
            data = json.loads(content)
            data["logs"] = read_last_logs()
            await websocket.send_json(data)
    except Exception as e:
        print(f"Initial sync error (live): {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── Estado genérico ─────────────────────────────────────────────────────────
@app.get("/api/state")
async def get_state(mode: str = "sim"):
    return read_state(mode=mode)
