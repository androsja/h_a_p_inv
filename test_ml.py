import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para poder importar módulos de trading_bot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.engine.trading_engine import TradingEngine
from shared.strategy.models import OpenPosition
import pandas as pd

def test_get_net_pnl():
    print("="*50)
    print("🧪 INICIANDO PRUEBA DE NET PNL PARA APRENDIZAJE IA")
    print("="*50)
    
    # Instanciamos el motor "mock" (no enviará ordenes reales)
    engine = TradingEngine(symbol="AAPL", broker=None, is_mock=True, session_num=1)
    
    # Simulamos un trade de 1 acción de AAPL a $150.00
    qty = 1.0
    entry_price = 150.00
    
    # Creamos una posicion fantasma (Ghost)
    engine.position = OpenPosition(
        symbol="AAPL", entry_price=entry_price, qty=qty,
        stop_loss=140.0, take_profit=160.0,
        is_ghost=True
    )
    
    print(f"\n📈 ESCENARIO 1: AAPL sube $0.10 centavos")
    sell_price_1 = 150.10
    
    # Calculamos PnL Bruto (Lo que usaba antes)
    gross_pnl_1 = (sell_price_1 - entry_price) * qty
    print(f"   Ganancia Bruta: +${gross_pnl_1:.2f}")
    
    # Calculamos PnL Neto (Lo que usa AHORA para la IA)
    net_pnl_1 = engine._get_net_pnl(gross_pnl_1, qty, entry_price)
    print(f"   Ganancia Neta (Enviada a IA): ${net_pnl_1:.2f}")
    print(f"   ▶ Resultado IA: {'WIN (1)' if net_pnl_1 > 0 else 'LOSS (0)'}")
    
    if net_pnl_1 < 0 and gross_pnl_1 > 0:
        print("   ✅ ÉXITO: La IA aprenderá que este trade es malo a pesar del falso verde.")
        
    print(f"\n📈 ESCENARIO 2: AAPL sube $1.00 dólar")
    sell_price_2 = 151.00
    gross_pnl_2 = (sell_price_2 - entry_price) * qty
    print(f"   Ganancia Bruta: +${gross_pnl_2:.2f}")
    net_pnl_2 = engine._get_net_pnl(gross_pnl_2, qty, entry_price)
    print(f"   Ganancia Neta (Enviada a IA): ${net_pnl_2:.2f}")
    print(f"   ▶ Resultado IA: {'WIN (1)' if net_pnl_2 > 0 else 'LOSS (0)'}")
    
    if net_pnl_2 > 0 and gross_pnl_2 > 0:
        print("   ✅ ÉXITO: Un salto grande sí se registra como Win.")

if __name__ == "__main__":
    test_get_net_pnl()
