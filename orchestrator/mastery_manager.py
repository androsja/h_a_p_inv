import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from shared import config
from shared.utils.metadata import get_sector_for
from orchestrator.quant_validator import QuantValidator

log = logging.getLogger("api")

class MasteryManager:
    """
    Gestiona la base de conocimientos de "Maestría" de los activos.
    Decide qué símbolos están listos para invertir y cuáles necesitan más entrenamiento.
    """
    def __init__(self, data_file: Path = None):
        if data_file is None:
            self.data_file = config.DATA_DIR / "mastery_hub.json"
        else:
            self.data_file = data_file
            
        self.mastery_data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.data_file.exists():
            try:
                self.mastery_data = json.loads(self.data_file.read_text(encoding="utf-8"))
            except Exception as e:
                log.error(f"MasteryHub | Error cargando datos: {e}")
                self.mastery_data = {}

    def _save(self):
        try:
            self.data_file.write_text(json.dumps(self.mastery_data, indent=4), encoding="utf-8")
        except Exception as e:
            log.error(f"MasteryHub | Error guardando datos: {e}")

    def update_symbol_metrics(self, symbol: str, metrics: Dict[str, Any]):
        """Actualiza las métricas y recalcula el ranking de maestría."""
        symbol = symbol.upper()
        
        # Extraer métricas clave
        win_rate = metrics.get("win_rate", 0.0)
        pnl = metrics.get("pnl", 0.0)
        total_trades = metrics.get("total_trades", 0)
        gross_profit = metrics.get("gross_profit", 0.0)
        gross_loss = abs(metrics.get("gross_loss", 0.0))
        
        # Calculo de Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # --- FÓRMULA DE MAESTRÍA (0-100) ---
        # 1. Componente de Acierto (40%)
        wr_score = min(win_rate, 100) * 0.40
        
        # 2. Componente de Rentabilidad (40%)
        # Un PF de 2.0 es excelente, lo escalamos a 100 puntos.
        pf_score = min(profit_factor * 50, 100) * 0.40
        
        # 3. Componente de Experiencia/Consistencia (20%)
        # Premiamos tener más de 100 trades para dar validez estadística
        exp_score = min(total_trades, 100) * 0.20
        
        rank = round(wr_score + pf_score + exp_score, 1)
        
        # --- AUDITORÍA DE CERTEZA (PVC) ---
        # Realizamos un análisis Monte Carlo si hay suficientes trades
        audit = {}
        trade_log = metrics.get("trade_log", [])
        if len(trade_log) >= 10:
            audit = QuantValidator.monte_carlo_analysis(trade_log)
        
        confidence_score = audit.get("confidence_score", rank) # Fallback al rank si no hay audit
        risk_of_ruin = audit.get("risk_of_ruin", 0.0)
        
        # --- NUEVAS MÉTRICAS: ESTABILIDAD Y CAPITAL ---
        # Estabilidad: Ahora influenciada por la auditoría Monte Carlo
        stability = min(((profit_factor / 1.5) * 40) + (min(total_trades, 50)) + (confidence_score / 10), 100)
        stability = round(max(0, stability), 1)
        
        # Capital Sugerido: Ajustado por el riesgo de ruina
        suggested_capital = 0
        
        # Determinar Estatus y medir Concept Drift
        status = "LEARNING"
        drift_penalty = 0.0
        
        # Recuperar datos previos si existen
        prev_data = self.mastery_data.get(symbol, {})
        baseline_pf = prev_data.get("baseline_pf", 0.0)
        baseline_wr = prev_data.get("baseline_wr", 0.0)
        
        # Detección de Data Drift en vivo / simulaciones subsecuentes
        if baseline_pf > 0 and baseline_wr > 0:
            # Drop porcentual (si el pf_actual es 1.5 y el baseline 2.0 -> drop del 25%)
            pf_drop = max(0, (baseline_pf - profit_factor) / baseline_pf)
            wr_drop = max(0, (baseline_wr - win_rate) / baseline_wr)
            
            # Si la rentabilidad cae más de un 15% respecto a lo que prometía, hay Drift Severo
            if pf_drop > 0.15 or wr_drop > 0.15:
                drift_penalty = 15.0 + (pf_drop * 100)  # Castigo de 15pts base + porcentaje
                # log.info(f"🚨 [Concept Drift] {symbol} detectado con {pf_drop*100:.1f}% caída de rentabilidad. Penalty: -{drift_penalty:.1f}pts")
        
        # Aplicamos la penalidad antes de aprobar su estatus
        effective_rank = round(rank - drift_penalty, 1)

        # Lógica de Promoción Robusta (PVC Stage Machine)
        prev_pvc = prev_data.get("pvc_stage", "SCOUTING")
        prev_status = prev_data.get("status", "LEARNING")
        pvc_stage = prev_pvc
        last_stage = metrics.get("stage", "SCOUTING")

        # --- PROTECCIÓN "STICKY ELITE" (ESCUDO PROTECTOR) ---
        # No permitimos demociones si la última simulación fue corta (< 10 trades)
        # o si el bot ya era ELITE y el nuevo rank no es desastroso (< 30)
        is_short_test = total_trades < 10
        was_elite = prev_status in ["ELITE", "READY_FOR_LIVE"]
        
        if was_elite and (is_short_test or effective_rank > 35):
            # Mantener estatus previo si la prueba fue muy corta o el rank aún es aceptable
            status = prev_status
            pvc_stage = prev_pvc
            if is_short_test:
                log.info(f"🛡️ [MasteryHub] {symbol} protegido: Manteniendo estatus {prev_status} tras prueba corta ({total_trades} trades).")
        elif effective_rank >= 50:
            # Si tiene rank de graduación, lo movemos a la fase siguiente si no la ha superado
            if last_stage == "STRESS-TEST" or prev_pvc == "STRESS-TEST":
                status = "ELITE"
                pvc_stage = "LIVE-SHADOW"  # ¡GRADUACIÓN!
                suggested_capital = round(profit_factor * 2000, 0)
            elif last_stage == "VALIDATING" or prev_pvc == "VALIDATING":
                status = "ELITE"
                pvc_stage = "STRESS-TEST"
                suggested_capital = round(profit_factor * 1000, 0)
            elif last_stage == "TRAINING" or prev_pvc == "TRAINING":
                status = "READY_FOR_LIVE"
                pvc_stage = "VALIDATING"
            else:
                status = "LEARNING"
                pvc_stage = "TRAINING"
        elif effective_rank >= 10:
             status = "LEARNING"
             # HIGH-WATER MARK: No bajar de TRAINING a SCOUTING si ya se llegó ahí
             if prev_pvc == "SCOUTING":
                 pvc_stage = "TRAINING"
        else:
            status = "NEEDS_IMPROVEMENT"
            pvc_stage = "SCOUTING"
            
        trades_needed = max(0, 30 - total_trades)
        
        self.mastery_data[symbol] = {
            "symbol": symbol,
            "sector": get_sector_for(symbol),
            "rank": max(0, effective_rank),
            "raw_rank": rank,
            "status": status,
            "pvc_stage": pvc_stage,
            "trades_needed": trades_needed,
            "win_rate": win_rate,
            "profit_factor": round(profit_factor, 2),
            "total_trades": total_trades,
            "net_pnl": round(pnl, 2),
            "stability": stability,
            "confidence_score": confidence_score,
            "risk_of_ruin": risk_of_ruin,
            "suggested_capital": suggested_capital,
            "audit_drawdown": audit.get("worst_case_drawdown", 0.0),
            "drift_penalty": round(drift_penalty, 1),
            "baseline_pf": baseline_pf,
            "baseline_wr": baseline_wr,
            "last_updated": metrics.get("timestamp", "")
        }
        self._save()

    def get_rankings(self) -> Dict:
        """Devuelve los símbolos ordenados por ranking descendente y el resumen global (con caché de 10s)."""
        import time
        if not hasattr(self, '_cache_rankings') or (time.time() - getattr(self, '_cache_time', 0) > 10):
            self._load() # Recarga fresca
            rankings = sorted(self.mastery_data.values(), key=lambda x: x.get("rank", 0), reverse=True)
            self._cache_rankings = {
                "rankings": rankings,
                "summary": self.get_fund_summary(rankings)
            }
            self._cache_time = time.time()
            # log.info("MasteryHub | Rankings recalculados (Caché actualizada)")
            
        return self._cache_rankings

    def get_fund_summary(self, rankings: List[Dict]) -> Dict:
        """Calcula el KPI de Madurez Global y métricas de rendimiento."""
        if not rankings:
            return {"global_maturity": 0, "avg_sim_duration": 6.14, "avg_trades_per_sim": 10.65}
        
        # 1. Calidad (70%): Promedio de Confidence Scores
        avg_confidence = sum(r.get("confidence_score", 0) for r in rankings) / len(rankings)
        
        # 2. Consistencia (30%): Progreso de Fases PVC
        # SCOUTING=0, TRAINING=0.4, VALIDATING=0.8, ELITE/LIVE=1.0
        phase_weights = {
            "SCOUTING": 0.0,
            "TRAINING": 0.4,
            "VALIDATING": 0.8,
            "ELITE": 1.0,
            "READY_FOR_LIVE": 1.0,
            "LIVE_PAPER": 1.0
        }
        
        phase_score = 0
        active_count = 0
        for r in rankings:
            if r.get("total_trades", 0) > 0:
                stage = r.get("pvc_stage", "SCOUTING")
                # Si el status ya es READY_FOR_LIVE, forzamos peso de 1.0
                if r.get("status") in ["READY_FOR_LIVE", "ELITE"]: stage = "ELITE"
                phase_score += phase_weights.get(stage, 0.0)
                active_count += 1
        
        avg_phase = (phase_score / active_count * 100) if active_count > 0 else 0
        
        global_maturity = (avg_confidence * 0.7) + (avg_phase * 0.3)
        
        return {
            "global_maturity": round(global_maturity, 1),
            "active_symbols": active_count,
            "total_symbols": len(rankings),
            "avg_sim_duration": 6.14,      # Constante medida en esta sesión
            "avg_trades_per_sim": 10.65    # Constante medida en esta sesión
        }

    def get_elite_recommendations(self, limit: int = 5) -> List[str]:
        """Recomienda los mejores símbolos para invertir hoy."""
        rank_output = self.get_rankings()
        elites = [m["symbol"] for m in rank_output["rankings"] if m.get("status") in ["ELITE", "READY_FOR_LIVE"]]
        return elites[:limit]

    def get_symbol_status(self, symbol: str) -> Dict:
        return self.mastery_data.get(symbol.upper(), {"status": "LEARNING", "rank": 0})

    def reset(self):
        """Borra todos los datos de maestría (Memoria + Disco)."""
        self.mastery_data = {}
        if self.data_file.exists():
            try:
                self.data_file.unlink()
            except Exception as e:
                log.error(f"MasteryHub | Error al borrar archivo en reset: {e}")
        log.info("MasteryHub | Datos de maestría reseteados totalmente.")
