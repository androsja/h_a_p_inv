# 🧠 Blueprint Maestro: Arquitectura Detallada de Hapi

Este documento es el manual técnico definitivo del sistema Hapi, diseñado para explicar paso a paso cómo funciona el cerebro detrás de tus inversiones. Está estructurado para ser procesado por una IA de generación de contenido (como NotebookLM) para crear un podcast profundo y detallado.

---

## 🏗️ 1. GOBERNANZA DE RECURSOS: El Guardián del PC
Antes de que cualquier algoritmo se ejecute, el sistema debe asegurar la estabilidad del hardware.

### Flujo de Trabajo en `DockerManager`:
1.  **Lectura de Salud:** La función `get_resource_health()` invoca a la librería `psutil` para obtener los porcentajes de CPU y RAM.
2.  **Cálculo de "Slots":** La función `get_available_slots()` aplica una restricción de seguridad: `RAM_Total - 1024MB`. Esto garantiza que tu Mac siempre tenga 1GB libre para el sistema operativo.
3.  **Decisión de Lanzamiento:** Si el cálculo devuelve un número superior a 0, el Orquestador tiene "luz verde" para lanzar una nueva simulación.

---

## 📈 2. EL CURRÍCULUM ACADÉMICO: El Sistema de Ascensos
El bot no opera en "Live" hasta que se gradúa en simulaciones extremas.

| Fase | Nombre del Estado | Periodo de Datos | Criterio de Éxito |
| :--- | :--- | :--- | :--- |
| **0** | **SIN DATOS** | N/A | Lanzar fase 1 inmediatamente. |
| **1** | **APRENDIENDO** | Ene 2024 - Dic 2024 | PnL > $0 y Profit Factor > 1.0. |
| **2** | **VALIDANDO** | Ene 2025 - Actualidad | Sobrevivir a datos desconocidos con rentabilidad. |
| **3** | **GRADUADO** | En cola para LIVE | Certificación técnica completa. |

---

## 🔍 3. ANATOMÍA DE UNA OPERACIÓN: Paso a Paso
¿Qué sucede exactamente desde que el mercado se mueve hasta que ganas dinero? Aquí está la cronología exacta de una señal:

### Paso 1: Generación de Señal (`indicators.py` -> `analyze()`)
El motor técnico observa el precio cada minuto.
*   **Trigger:** "El precio acaba de tocar la EMA 20 en una tendencia alcista".
*   **Acción:** Se genera una propuesta interna de **BUY signal**.

### Paso 2: Evaluación por el Juez IA (`NeuralTradeFilter` -> `predict()`)
El sistema invoca a la Red Neuronal antes de comprometer capital.
1.  **Ingeniería de Características:** Se capturan las 13 variables del mercado (ver sección 4).
2.  **Procesamiento MLP:** Los datos pasan por las capas neuronales (32 → 16 → 8).
3.  **Decisión:** Si la probabilidad es menor al threshold (60%), el trade se veta. Si es mayor, se solicita permiso de fondos.

### Paso 3: Solicitud de Fondos (`BankManager` -> `process_transaction()`)
El bot llama al Banco Central del Orquestador.
*   **Validación:** El Banco verifica si hay efectivo disponible (`available_cash`).
*   **Aprobación:** Si hay fondos, se restan del saldo global y se autoriza la "compra" en el simulador.

### Paso 4: Ejecución y Gestión de Riesgo (`TradingEngine` -> `run_trade()`)
El bot abre la posición e inicia su vigilancia.
*   **Trailing Stop Dinámico:** El bot mueve el Stop Loss hacia arriba a medida que el precio sube para asegurar que una ganancia no se convierta en pérdida.

### Paso 5: Feedback y Ciclo de Aprendizaje (`NeuralTradeFilter` -> `fit()`)
Cuando el trade termina, sucede la "magia" del perfeccionamiento:
1.  **Recolección de Datos:** Se registra si el trade fue EXITOSO (1) o FALLIDO (0).
2.  **Online Learning:** La IA ajusta sus pesos internos basándose en este nuevo ejemplo.
3.  **Consolidación:** Cada 50 operaciones, la IA realiza un entrenamiento de refuerzo para adaptarse a la volatilidad actual.

---

## 🧬 4. LOS 13 OJOS DE LA IA (Features)
Para que el podcast sea detallado, estos son los nombres exactos de lo que la IA observa en cada operación:

1.  **`h_norm`**: Hora del día normalizada (de 0 a 1).
2.  **`vwap_dist_pct`**: Distancia porcentual al VWAP (Precio promedio).
3.  **`rsi`**: Índice de Fuerza Relativa (Indica si el activo está sobrecomprado).
4.  **`macd_hist`**: El momentum o "empuje" del mercado.
5.  **`atr_pct`**: Volatilidad relativa (qué tan grande es la "llama" del movimiento).
6.  **`vol_ratio`**: Comparación del volumen actual vs el promedio histórico.
7.  **`ema_spread_pct`**: Separación entre medias móviles (Fuerza de la tendencia).
8.  **`zscore_vwap`**: Qué tan atípico es el precio respecto a la campana de Gauss.
9.  **`regime_enc`**: El tipo de mercado (Tendencia vs Caos).
10. **`num_confirmations`**: Cuántos indicadores técnicos coinciden en la entrada.
11. **`adx`**: Fuerza direccional (¿El movimiento tiene energía?).
12. **`has_pattern`**: Detección de patrones de velas (como Martillos o Envolventes).
13. **`is_adx_rising`**: ¿La fuerza del mercado está subiendo o bajando?

---

## 🛡️ 5. ARQUITECTURA DEL "CEREBRO GLOBAL"
Hapi utiliza un modelo de **Inteligencia Colectiva**.

*   **Archivo Maestro:** `neural_GLOBAL.joblib`.
*   **Funcionamiento:** Todas las simulaciones (AAPL, AMZN, etc.) escriben en el mismo cerebro. 
*   **Beneficio:** Si el bot de **NVDA** pierde dinero comprando a las 4:00 PM (hora peligrosa), el bot de **AMZN** lo sabe al instante y evitará disparar a esa misma hora. Es una sabiduría compartida.

---

> [!IMPORTANT]
> **Narrativa de Cierre para el Podcast:**
> Hapi no es un programa estático; es un organismo digital en constante evolución. El **Auto-Perfeccionador** actúa como el ADN del sistema, asegurando que solo las estrategias más fuertes sobrevivan a través de un proceso de **Darwinismo Algorítmico**. Cada trade fallido no es una pérdida, es una **lección** que se graba en el Cerebro Global para siempre. Mientras tú duermes, tu PC está fabricando un "Trader Perfecto".
