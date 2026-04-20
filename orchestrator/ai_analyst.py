import os
import asyncio
from typing import Dict, Any
from google import genai
from shared import config

class HapiAIAnalyst:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("⚠️ [AI Analyst] No GEMINI_API_KEY found in environment.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
        
        self.model_name = "gemini-flash-latest" # Stable latest alias

    async def analyze_simulation(self, symbol: str, bot_data: Dict[str, Any], mastery_data: Dict[str, Any]) -> str:
        """
        Generates a strategic audit for a running simulation bot.
        """
        if not self.client:
            return "❌ Error: GEMINI_API_KEY no configurada. No se puede realizar el análisis."

        # Assembler context
        stage = bot_data.get("stage", "UNKNOWN")
        trades = bot_data.get("trades", 0)
        pnl_net = bot_data.get("pnl_net", 0.0)
        pnl_gross = bot_data.get("pnl_gross", 0.0)
        win_rate = bot_data.get("win_rate", 0.0)
        progress = bot_data.get("progress", 0.0)
        
        # Mastery metrics
        rank = mastery_data.get("rank", 0)
        status = mastery_data.get("status", "N/A")
        pf = mastery_data.get("profit_factor", 1.0)
        drawdown = mastery_data.get("max_drawdown", 0.0)

        prompt = f"""
        Actúa como un Auditor Senior de Estrategiasititativas (Quant Auditor). 
        Analiza el rendimiento actual de un bot de trading de Hapi que está operando el símbolo: {symbol}.

        DATOS DE LA SIMULACIÓN:
        - Etapa Actual: {stage}
        - Progreso: {progress}%
        - Operaciones Realizadas: {trades}
        - PnL Neto: ${pnl_net:.2f} (Bruto: ${pnl_gross:.2f})
        - Win Rate: {win_rate:.1f}%
        - Profit Factor: {pf:.2f}
        - Max Drawdown: {drawdown:.1f}%
        - Rango de Maestría: {rank}/100 ({status})

        TAREA:
        Proporciona un veredicto estratégico conciso pero profundo. El usuario verá esto en el dashboard.
        Usa un tono profesional, directo y basado en datos.

        ESTRUCTURA DE RESPUESTA (Usa Markdown):
        1. **Veredicto Ejecutivo**: (Saludable, Prometedor, Riesgoso, o Tóxico).
        2. **Análisis de Métricas**: ¿Los trades son suficientes para confiar? ¿El Win Rate es realista?
        3. **Detección de Riesgos**: (Ej: Overfitting, Slippage excesivo, falta de volatilidad).
        4. **Potencial de 'Elite'**: ¿Recomiendas que este bot siga hasta ser Elite?
        
        Mantén la respuesta en ESPAÑOL y que sea de lectura rápida (máximo 200 palabras).
        """

        try:
            # Running in executor to not block async loop if SDK is synchronous
            def call_gemini():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, call_gemini)
            return response.text
        except Exception as e:
            # Manejo específico de cuota (429) o error técnico raw
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                return f"ERROR_429: {err_msg}"
            return f"ERROR: {err_msg}"

    async def analyze_global(self, bots_active: list, completed_recent: list, bank_data: dict, ai_central_stats: dict) -> str:
        """
        Generates a holistic report of the entire trading operation.
        """
        if not self.client:
            return "❌ Error: API Key no configurada."

        summary_bots = []
        for b in bots_active[:10]: # Limit context to avoid token bloat
            summary_bots.append(f"- {b['symbol']}: {b.get('stage')} | PnL: ${b.get('pnl_net', 0):.2f} | WR: {b.get('win_rate', 0):.1f}%")
        
        bots_text = "\n".join(summary_bots)
        
        prompt = f"""
        Actúa como el Director (CEO) de un Fondo de Inversión Quant. 
        Tu objetivo es dar un reporte de estado "Flash" al dueño del fondo.

        ESTADO ACTUAL DEL FONDO:
        - Balance en Banco: ${bank_data.get('available_cash', 0):.2f}
        - PnL Total del Banco: ${bank_data.get('total_pnl', 0):.2f}
        - Bots Activos en este momento: {len(bots_active)}
        - Muestras en IA Central: {ai_central_stats.get('total_samples', 0)}
        - Precisión IA Central: {ai_central_stats.get('model_accuracy', 0)}%

        DETALLE DE BOTS (Top 10):
        {bots_text}

        TAREA:
        Escribe un reporte de gestión (Auditoría Global).
        1. **Resumen de Salud del Fondo**: ¿Vamos ganando o perdiendo? ¿La IA central está madura?
        2. **Análisis de la Flota**: ¿Los bots están en etapas tempranas o avanzadas? ¿Hay algún símbolo que brille o preocupe?
        3. **Nivel de Riesgo**: ¿Estamos sobre-expuestos o el capital está bien protegido?
        4. **Próximo Paso Estratégico**: ¿Lanzar más bots, limpiar el banco, o dejar que la IA aprenda más?

        Mantén un tono de "Reporte de Inteligencia" de alto nivel. 
        Sé directo. Máximo 250 palabras. ESPAÑOL.
        """

        try:
            def call_gemini():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, call_gemini)
            return response.text
        except Exception as e:
            # Manejo específico de cuota (429) o error técnico raw
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                return f"ERROR_429: {err_msg}"
            return f"ERROR: {err_msg}"

# Singleton
hapi_ai = HapiAIAnalyst()
