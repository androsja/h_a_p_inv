import json
from pathlib import Path
from typing import List, Dict, Optional
from shared import config

QUEUE_FILE = config.DATA_DIR / "launch_queue.json"

class QueueManager:
    def __init__(self):
        self.queue_file = QUEUE_FILE
        self.queue: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self.queue_file.exists():
            try:
                return json.loads(self.queue_file.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self):
        try:
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            self.queue_file.write_text(json.dumps(self.queue, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ [QueueManager] Error guardando cola: {e}")

    def add_to_queue(self, request_data: Dict) -> bool:
        """Añade un lanzamiento a la cola si no está ya presente."""
        symbol = request_data.get("symbol", "").upper()
        if not symbol:
            return False

        # Evitar duplicados en cola
        if any(item.get("symbol") == symbol for item in self.queue):
            return False

        self.queue.append(request_data)
        self._save()
        return True

    def pop_next(self) -> Optional[Dict]:
        """Obtiene y elimina el primer elemento de la cola."""
        if not self.queue:
            return None
        
        item = self.queue.pop(0)
        self._save()
        return item

    def get_queue(self) -> List[Dict]:
        """Retorna la lista actual de la cola."""
        return self.queue

    def remove_from_queue(self, symbol: str) -> bool:
        """Elimina un símbolo específico de la cola."""
        symbol = symbol.upper()
        initial_len = len(self.queue)
        self.queue = [item for item in self.queue if item.get("symbol") != symbol]
        
        if len(self.queue) != initial_len:
            self._save()
            return True
        return False
