import pytest
from shared.utils.neural_filter import RPCNeuralTradeFilter

# Mockup URL para tests falsos
TEST_URL = "http://localhost:9000"

def test_rpc_freeze_mechanics():
    """Valida la mecánica interna de congelamiento de la red para evitar Overfitting en cascada."""
    rpc = RPCNeuralTradeFilter(TEST_URL)
    assert not rpc.is_frozen
    
    rpc.freeze()
    assert rpc.is_frozen
    
    rpc.unfreeze()
    assert not rpc.is_frozen

def test_rpc_fallback_returns():
    """Valida que si el cerebro falla por timeout, no destruye el bot, devuelve neutro."""
    # Le pasamos una URL falsa que nadie atiende
    bad_rpc = RPCNeuralTradeFilter("http://localhost:9999")
    
    proba, reason = bad_rpc.predict([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Debe devolver un fallback neutro (50% = 0.5)
    assert proba == 0.5
    assert "Error" in reason
