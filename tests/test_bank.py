import pytest
import asyncio
from orchestrator.bank import BankManager

@pytest.fixture
def bank():
    # Inicializar con $10,000 para las pruebas
    return BankManager(initial_cash=10000.0)

def test_bank_initial_balance(bank):
    assert bank.available_cash == 10000.0

def test_bank_successful_buy(bank):
    success, msg = bank.process_transaction("AAPL", 150.0, "buy")
    assert success is True
    assert bank.available_cash == 9850.0

def test_bank_insufficient_funds(bank):
    success, msg = bank.process_transaction("AAPL", 11000.0, "buy")
    assert success is False
    assert bank.available_cash == 10000.0

def test_bank_successful_sell(bank):
    # En una venta, el banco suma el dinero ingresado
    success, msg = bank.process_transaction("AAPL", 150.0, "sell")
    assert success is True
    assert bank.available_cash == 10150.0

import threading

def test_bank_concurrency(bank):
    """
    Test de concurrencia: simula que 10 instancias intentan comprar/vender 
    exactamente al mismo milisegundo (Race Condition test).
    El BankManager usa threading.Lock así que debería poder soportarlo sin corromper el saldo.
    """
    def worker(monto, tipo):
        bank.process_transaction("HAPI", monto, tipo)

    threads = []
    # 5 compras de 100 y 5 ventas de 100
    for _ in range(5):
        t1 = threading.Thread(target=worker, args=(100.0, "buy"))
        t2 = threading.Thread(target=worker, args=(100.0, "sell"))
        threads.extend([t1, t2])
        t1.start()
        t2.start()

    for t in threads:
        t.join()

    # Como compramos 500 y vendimos 500, el saldo final debe ser EXACTAMENTE 10,000.0
    assert bank.available_cash == 10000.0
