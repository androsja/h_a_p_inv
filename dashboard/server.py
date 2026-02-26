"""
dashboard/server.py ─ Servidor FastAPI + WebSocket para el dashboard en tiempo real.

Lee el estado del bot desde /app/data/state.json (volumen compartido con el bot)
y lo empuja a todos los clientes conectados vía WebSocket cada segundo.
También sirve el dashboard HTML estático.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATE_FILE = Path("/app/data/state.json")
LOG_FILE   = Path("/app/logs/trading_bot.log")
HTML_FILE  = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="Hapi Bot Dashboard")

# ─── Sirve archivos estáticos ─────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Manager de conexiones WebSocket ─────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── Lectura del estado del bot ───────────────────────────────────────────────
def read_state() -> dict:
    state = {}
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                state = json.load(f)
    except Exception:
        state = {"status": "starting", "timestamp": datetime.utcnow().isoformat()}

    # Inyectar símbolos forzados (si hay) para que el UI genere las pestañas
    cmd_file = Path("/app/data/command.json")
    if cmd_file.exists():
        try:
            with open(cmd_file) as f:
                cmds = json.load(f)
                state["force_symbols"] = cmds.get("force_symbols", [])
        except: pass

    return state


def read_last_logs(n: int = 50) -> list[str]:
    """Lee las últimas n líneas del log."""
    try:
        if LOG_FILE.exists():
            lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
            return lines[-n:]
    except Exception:
        pass
    return []


# ─── Broadcaster en background ────────────────────────────────────────────────
async def state_broadcaster():
    """Tarea en background: empuja el estado a todos los clientes cada 1 segundo."""
    while True:
        state = read_state()
        state["logs"] = read_last_logs(30)
        await manager.broadcast(state)
        await asyncio.sleep(1)


@app.on_event("startup")
async def startup():
    asyncio.create_task(state_broadcaster())


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    content = HTML_FILE.read_text(encoding="utf-8")
    return HTMLResponse(content=content, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Mantener la conexión viva esperando mensajes del cliente
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/state")
async def get_state():
    return read_state()


@app.post("/api/set_symbol")
async def set_symbol(symbol: str):
    import json
    command_file = Path("/app/data/command.json")
    try:
        data = {}
        if command_file.exists():
            with open(command_file) as f:
                data = json.load(f)
        data["force_symbol"] = symbol
        with open(command_file, "w") as f:
            json.dump(data, f)
        return {"status": "success", "symbol": symbol}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/reset_all")
async def reset_all():
    import json
    command_file = Path("/app/data/command.json")
    try:
        data = {}
        if command_file.exists():
            with open(command_file) as f:
                data = json.load(f)
        data["reset_all"] = True
        data["force_paper_trading"] = False
        data["force_symbols"] = []
        data["force_symbol"] = "AUTO"
        with open(command_file, "w") as f:
            json.dump(data, f)
        
        # Limpiar archivos en background
        for f in ["/app/data/backtest_results.json", "/app/data/ml_dataset.csv", "/app/data/state.json"]:
            try:
                if Path(f).exists():
                    Path(f).unlink()
            except: pass
            
        # Limpiar logs
        log_file = Path("/app/logs/trading_bot.log")
        try:
            if log_file.exists():
                with open(log_file, 'w') as f:
                    pass # Truncate to 0 bytes
        except: pass
            
        return {"status": "success", "message": "Reseteando bot..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/results")
async def get_results():
    import json
    res_file = Path("/app/data/backtest_results.json")
    try:
        if res_file.exists():
            with open(res_file) as f:
                return json.load(f)
        return []
    except Exception:
        return []

@app.post("/api/train_ai")
async def train_ai():
    import subprocess
    try:
        result = subprocess.run(["python3", "/app/train_ai.py"], capture_output=True, text=True)
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/config")
async def get_config():
    import json
    assets_file = Path("/app/assets.json")
    try:
        if assets_file.exists():
            with open(assets_file) as f:
                data = json.load(f)
                active_symbols = [a for a in data.get('assets', []) if a.get('enabled', True)]
                return {"total_symbols": len(active_symbols)}
        return {"total_symbols": 46}
    except Exception:
        return {"total_symbols": 46}

@app.get("/api/active_symbols")
async def get_active_symbols():
    import json
    assets_file = Path("/app/assets.json")
    try:
        if assets_file.exists():
            with open(assets_file) as f:
                data = json.load(f)
                active = [a.get('symbol') for a in data.get('assets', []) if a.get('enabled', True)]
                return {"status": "success", "symbols": active}
        return {"status": "error", "symbols": []}
    except Exception as e:
        return {"status": "error", "symbols": []}

class DepositRequest(BaseModel):
    amount: float

@app.post("/api/bank/deposit")
async def bank_deposit(req: DepositRequest):
    import json
    state_file = Path("/app/data/state.json")
    try:
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            data["available_cash"] = data.get("available_cash", 10000.0) + req.amount
            with open(state_file, "w") as f:
                json.dump(data, f)
            return {"status": "success", "message": f"Deposited ${req.amount:.2f}", "new_balance": data["available_cash"]}
        return {"status": "error", "message": "State file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/bank/withdraw")
async def bank_withdraw(req: DepositRequest):
    import json
    state_file = Path("/app/data/state.json")
    try:
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            # Hapi fixed withdrawal fee $4.99
            withdrawal_fee = 4.99
            total_deduction = req.amount + withdrawal_fee
            
            if data.get("available_cash", 10000.0) >= total_deduction:
                data["available_cash"] -= total_deduction
                with open(state_file, "w") as f:
                    json.dump(data, f)
                return {"status": "success", "message": f"Withdrew ${req.amount:.2f} (Fee: ${withdrawal_fee:.2f})", "new_balance": data["available_cash"]}
            else:
                return {"status": "error", "message": "Insufficient funds including $4.99 withdrawal fee."}
        return {"status": "error", "message": "State file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PaperTradeRequest(BaseModel):
    symbols: str  # Comma separated list like "TSLA,AAPL,NVDA"

@app.post("/api/paper_trade_start")
async def paper_trade_start(req: PaperTradeRequest):
    import json
    command_file = Path("/app/data/command.json")
    try:
        data = {}
        if command_file.exists():
            with open(command_file) as f:
                data = json.load(f)
        
        # Parse symbols
        symbol_list = [s.strip().upper() for s in req.symbols.split(",") if s.strip()]
        if not symbol_list:
            return {"status": "error", "message": "No se proporcionaron símbolos válidos."}

        data["force_paper_trading"] = True
        data["force_symbols"] = symbol_list
        data["force_symbol"] = symbol_list[0] # Start with the first one
        data["reset_all"] = True
        
        with open(command_file, "w") as f:
            json.dump(data, f)
            
        return {"status": "success", "message": f"Iniciando Live Paper para: {', '.join(symbol_list)}. Reiniciando Bot."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/history")
async def get_history(symbol: str, interval: str = "5m", period: str = "5d"):
    import yfinance as yf
    try:
        df = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        if df.empty:
            # Try 1m
            df = yf.download(tickers=symbol, period="1d", interval="1m", progress=False)
        
        if df.empty:
            return {"status": "error", "message": "No hay datos para este símbolo."}

        # Convertir MultiIndex a simple si es necesario
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        candles = []
        for ts, row in df.iterrows():
             candles.append({
                 "time": int(ts.timestamp()),
                 "open": float(row["Open"]),
                 "high": float(row["High"]),
                 "low": float(row["Low"]),
                 "close": float(row["Close"]),
                 "volume": int(row["Volume"])
             })
        return {"status": "success", "symbol": symbol, "candles": candles}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/paper_trade_stop")
async def paper_trade_stop():
    import json
    command_file = Path("/app/data/command.json")
    try:
        data = {}
        if command_file.exists():
            with open(command_file) as f:
                data = json.load(f)
        
        data["force_paper_trading"] = False
        data["force_symbols"] = []
        data["force_symbol"] = "AUTO"
        data["reset_all"] = True
        
        with open(command_file, "w") as f:
            json.dump(data, f)
            
        return {"status": "success", "message": "Live Paper Trading detenido. Volviendo a la Simulación Global."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
