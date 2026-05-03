# 🎯 Filosofía de Trading: El Francotirador Equilibrado

Este documento resume la estrategia oficial del bot tras la optimización del 25 de Abril de 2026.

## 核心 (Core Objective)
**"Ganar mucho, perder poco, y disparar cuando hay ventaja."**

El sistema ha pasado de ser un "Muro de Acero" (que no operaba nunca) a un "Francotirador Equilibrado". Aceptamos que el trading implica pérdidas pequeñas, pero exigimos que el bot capture las grandes tendencias del mercado.

## 🛠️ Configuración de Filtros Técnicos

| Filtro | Valor Actual | Propósito |
| :--- | :--- | :--- |
| **ADX (Fuerza)** | `> 20` | Asegura que haya movimiento, pero permite entrar en tendencias constantes (no solo parabólicas). |
| **EMA 200** | Pendiente Positiva | Solo compramos si la tendencia de largo plazo es alcista. |
| **Oracle Sentinel** | `0.51` | Umbral dinámico que usa la Red Neuronal para bloquear trades de alta probabilidad de fracaso. |
| **Doble Consenso IA** | RF + Neural | Solo bloqueamos si AMBOS modelos de IA odian el trade. Si uno ve oportunidad, disparamos. |
| **RSI** | `< 70` | Evita comprar en el pico máximo de euforia (sobrecompra). |

## 🛡️ Gestión de Riesgo (Money Management)
1.  **Stop Loss Dinámico (ATR):** Las pérdidas se cortan automáticamente basándose en la volatilidad real.
2.  **Trailing Stop:** Una vez que el trade es ganador, el stop "persigue" al precio para asegurar ganancias y evitar que un trade ganador se vuelva perdedor.
3.  **Aceptación de Pérdida:** Entendemos que para ganar un +40% en un activo como MU, el bot podría intentar 2 o 3 entradas fallidas que resulten en pérdidas de -$1 o -$2. Esto es parte del costo de hacer negocios.

## 📈 Comportamiento Esperado
*   **Mercado Lateral:** El bot operará poco o nada. Las pérdidas serán mínimas.
*   **Mercado Tendencial:** El bot debe ser capaz de entrar y mantener la posición mientras la tendencia sea fuerte.

---
*Documento generado por Antigravity AI - 25 de Abril de 2026*
