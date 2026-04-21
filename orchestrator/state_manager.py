import asyncio
import json
import time
from typing import Dict, Any, List
from fastapi import WebSocket

class StateManager:
    def __init__(self):
        # Almacenamiento central del estado de cada símbolo/bot
        self.global_state: Dict[str, Dict[str, Any]] = {}
        # Conexiones activas del Dashboard Frontend
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        
        # ⏱️ Registro de tiempo del último latido por símbolo
        self.last_updates: Dict[str, float] = {}
        
        # 🧪 SEMILLA DE MEMORIA: Cargar estado previo desde disco al arrancar
        try:
            from shared import config
            state_file = config.DATA_DIR / "state_live.json"
            if state_file.exists():
                import json
                data = json.loads(state_file.read_text(encoding="utf-8"))
                # state_live.json tiene estructura multi-key: {"MU": {...}, "ACN": {...}}
                self.global_state = data
                print(f"📡 [StateManager] Memoria sembrada con {len(data)} símbolos desde state_live.json")
        except Exception as e:
            print(f"⚠️ [StateManager] No se pudo cargar memoria inicial: {e}")
        
        # 🔄 Iniciar ciclo de persistencia a disco (para que el Dashboard externo vea los cambios)
        asyncio.create_task(self._persist_loop())

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            # Enviar el estado acumulado inmediatamente al conectarse
            try:
                await websocket.send_json(self.global_state)
            except Exception:
                pass

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def update_state(self, symbol: str, new_data: Dict[str, Any]):
        async with self._lock:
            # new_data puede tener DOS estructuras según el tipo de envío del worker:
            # a) Plano (nuevo módulo): {"status": "running", "symbol": "NVDA", ...}
            # b) Multi-key (legacy del state_writer): {"NVDA": {...}, "_main": {...}}
            
            is_flat = isinstance(new_data, dict) and not any(
                isinstance(v, dict) for v in new_data.values()
            )

            if is_flat or "symbol" in new_data or "status" in new_data:
                # Payload plano → indexar directamente bajo el símbolo recibido
                if symbol not in self.global_state:
                    self.global_state[symbol] = {}
                for k, v in new_data.items():
                    self.global_state[symbol][k] = v
                
                # Actualizar latido
                self.last_updates[symbol] = time.time()
                
                # Actualizar _main con el último símbolo que reportó
                self.global_state["_main"] = self.global_state[symbol]
            else:
                # Payload multi-key → iterar y guardar por clave
                for key, obj in new_data.items():
                    if isinstance(obj, dict):
                        self.global_state[key] = obj
                        self.last_updates[key] = time.time()
                        if key != "_main":
                            self.global_state["_main"] = obj
        # Difundir cambios de forma asíncrona (no bloqueante)
        asyncio.create_task(self.broadcast())

    async def broadcast(self):
        # Capturar un snapshot rápido
        async with self._lock:
            snapshot = dict(self.global_state)
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(snapshot)
            except Exception:
                disconnected.append(connection)
                
        # Limpiar conexiones muertas
        async with self._lock:
            for d in disconnected:
                if d in self.active_connections:
                    self.active_connections.remove(d)

    async def clear(self):
        async with self._lock:
            self.global_state.clear()
        # Difundir limpieza de forma asíncrona
        asyncio.create_task(self.broadcast())

    async def clear_symbol(self, symbol: str):
        """Limpia el estado en memoria de un símbolo específico."""
        async with self._lock:
            symbol_up = symbol.upper()
            if symbol_up in self.global_state:
                del self.global_state[symbol_up]
            # También limpiar de main si era el último
            main = self.global_state.get("_main", {})
            if main.get("symbol") == symbol_up:
                if "_main" in self.global_state:
                    del self.global_state["_main"]
        # Difundir y guardar
        asyncio.create_task(self.broadcast())
        await self._save_to_disk()

    async def _persist_loop(self):
        """Ciclo infinito para guardar el estado en disco cada pocos segundos."""
        while True:
            await asyncio.sleep(5) # Guardar cada 5 segundos
            await self._save_to_disk()

    async def _save_to_disk(self):
        """Persiste el global_state en los archivos JSON compartidos."""
        from shared import config
        async with self._lock:
            if not self.global_state:
                return
            snapshot = dict(self.global_state)
            
        try:
            # Guardar en state_sim.json (para el Dashboard externo)
            sim_path = config.DATA_DIR / "state_sim.json"
            sim_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
            
            # También en state_live.json para persistencia entre reinicios
            live_path = config.DATA_DIR / "state_live.json"
            live_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            # Silenciar errores de escritura frecuentes para no inundar logs
            pass
