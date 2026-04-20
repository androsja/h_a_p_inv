import random
import numpy as np
import logging
from typing import List, Dict

log = logging.getLogger("api")

class QuantValidator:
    """
    Auditor de grado institucional para validación de estrategias de trading.
    Implementa Monte Carlo, Walk-Forward y Risk of Ruin.
    """
    
    @staticmethod
    def monte_carlo_analysis(trades: List[Dict], iterations: int = 1000) -> Dict:
        """
        Simula 1000 mundos paralelos reordenando los trades.
        Calcula la probabilidad de quiebra y el drawdown máximo probable.
        """
        if not trades:
            return {"confidence": 0, "risk_of_ruin": 1.0}
            
        pnls = [t.get("pnl", 0) for t in trades]
        initial_balance = 10000 # Balance base para el test
        
        drawdowns = []
        final_balances = []
        
        for _ in range(iterations):
            shuffled = random.sample(pnls, len(pnls))
            balance = initial_balance
            peak = initial_balance
            max_dd = 0
            
            for p in shuffled:
                balance += p
                if balance > peak: peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                if dd > max_dd: max_dd = dd
            
            drawdowns.append(max_dd)
            final_balances.append(balance)
            
        avg_max_dd = np.mean(drawdowns)
        worst_dd = np.max(drawdowns)
        ruined = len([b for b in final_balances if b <= initial_balance * 0.5]) # Quiebra = perder el 50%
        
        risk_of_ruin = ruined / iterations
        confidence = (1.0 - risk_of_ruin) * 100
        
        return {
            "confidence_score": round(confidence, 1),
            "risk_of_ruin": round(risk_of_ruin * 100, 2),
            "avg_max_drawdown": round(avg_max_dd * 100, 2),
            "worst_case_drawdown": round(worst_dd * 100, 2)
        }

    @staticmethod
    def walk_forward_efficiency(is_results: Dict, oos_results: Dict) -> float:
        """
        Compara el rendimiento en el periodo de optimización vs el periodo de prueba.
        WFE = (OOS Annualized PnL / IS Annualized PnL).
        Un WFE > 50% es aceptable institucionalmente.
        """
        is_pnl = is_results.get("pnl", 0)
        oos_pnl = oos_results.get("pnl", 0)
        
        if is_pnl <= 0: return 0.0
        
        # Simplificado para este MVP: Ratio de rentabilidad relativa
        efficiency = (oos_pnl / is_pnl) * 100
        return round(min(efficiency, 150), 1) # Capeado a 150%
