import threading
import logging
from typing import Dict, List, Tuple
from shared.utils.metadata import get_sector_for

log = logging.getLogger("api")

class BankManager:
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.available_cash = initial_cash
        self.lock = threading.Lock()
        self.transactions = []
        
        # --- NUEVO: Gestión de Riesgo por Sector ---
        self.active_positions: Dict[str, float] = {} # symbol -> current_investment
        self.max_bots_per_sector = 10 # Relajado de 2 a 10 para permitir validación masiva por sectores
        
        # Mastery Hub reference (se inyectará desde main.py)
        self.mastery_mgr = None

    def process_transaction(self, symbol: str, amount: float, t_type: str) -> Tuple[bool, str]:
        """
        Procesa transacciones con filtro de diversificación y límites de capital.
        """
        symbol = symbol.upper()
        sector = get_sector_for(symbol)

        if amount < 0:
            return False, "Amount must be positive"

        with self.lock:
            if t_type == "buy":
                # 🛡️ 1. Verificar Saturación de Sector
                active_in_sector = [s for s in self.active_positions.keys() if get_sector_for(s) == sector]
                if len(active_in_sector) >= self.max_bots_per_sector:
                    log.warning(f"🏦 [Bank] Rechazado {symbol}: Sector {sector} saturado ({len(active_in_sector)} bots activos).")
                    return False, f"Sector {sector} concentration limit reached"

                # 🛡️ 2. Verificar Fondos Disponibles
                if self.available_cash < amount:
                    return False, f"Insufficient funds (Requested {amount:.2f}, Available {self.available_cash:.2f})"

                # 🛡️ 3. Lógica de "Dinamismo": Escalar según Maestría (Opcional si viene en el request)
                # Por ahora solo restamos y registramos
                self.available_cash -= amount
                self.active_positions[symbol] = amount
                self.transactions.append({"type": "buy", "symbol": symbol, "amount": amount, "sector": sector})
                return True, "Funds approved and reserved"

            elif t_type == "sell":
                # Regreso de fondos (Capi + PnL)
                self.available_cash += amount
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                
                self.transactions.append({"type": "sell", "symbol": symbol, "amount": amount, "sector": sector})
                return True, "Funds successfully deposited"
            
            else:
                return False, f"Invalid transaction type: {t_type}"

    def get_sector_exposure(self) -> Dict[str, int]:
        """Informa cuántos bots hay por cada sector."""
        exposure = {}
        with self.lock:
            for s in self.active_positions.keys():
                sec = get_sector_for(s)
                exposure[sec] = exposure.get(sec, 0) + 1
        return exposure

    def reset(self):
        """Resetea el banco al estado inicial."""
        with self.lock:
            self.available_cash = self.initial_cash
            self.transactions = []
            self.active_positions = {}
