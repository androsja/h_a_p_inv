import threading

class BankManager:
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.available_cash = initial_cash
        self.lock = threading.Lock()
        self.transactions = []

    def process_transaction(self, symbol: str, amount: float, t_type: str) -> tuple[bool, str]:
        """
        Processes a transaction atomically.
        t_type can be 'buy' (deducts cash) or 'sell' (adds cash).
        amount is strictly positive in both cases.
        """
        if amount < 0:
            return False, "Amount must be positive"

        with self.lock:
            if t_type == "buy":
                if self.available_cash >= amount:
                    self.available_cash -= amount
                    self.transactions.append({"type": "buy", "symbol": symbol, "amount": amount})
                    return True, "Funds approved and reserved"
                else:
                    return False, f"Insufficient funds (Requested {amount:.2f}, Available {self.available_cash:.2f})"
            elif t_type == "sell":
                # Returning funds to the bank (Capital + Profit/Loss)
                self.available_cash += amount
                self.transactions.append({"type": "sell", "symbol": symbol, "amount": amount})
                return True, "Funds successfully deposited"
            else:
                return False, f"Invalid transaction type: {t_type}"
