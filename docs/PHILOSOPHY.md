# 🧠 Filosofía de Entrenamiento Hapi (Walk-Forward Analysis)

Este documento detalla la lógica operativa y la mentalidad detrás de los ciclos de aprendizaje del sistema Hapi.

## 1. El Ciclo de Dos Fases (Dual-Phase Engine)

El sistema opera bajo una estructura de **Walk-Forward Analysis (WFA)** dividida en dos estados mentales distintos para la IA y el Oráculo:

### A. Fase de ESTUDIO (Cerebro Abierto) 📚
*   **Mentalidad:** Exploración y Aprendizaje.
*   **Objetivo:** Recolectar la mayor cantidad de datos posible sobre un mes específico del pasado.
*   **Comportamiento del Oráculo:** Actúa como un **Tutor**. Permite que la IA ejecute trades incluso si el Oráculo detecta riesgo moderado (aplicando penalizaciones de tamaño pero no bloqueos totales).
*   **Selectividad IA (0%):** En esta fase, es normal ver un 0% de selectividad. La IA "tiene permiso para equivocarse" para que los pesos neuronales puedan ajustarse correctamente.
*   **PnL:** Puede ser negativo. Es el "Costo de Matrícula" del aprendizaje.

### B. Fase de VALIDACIÓN (Examen Maestro) 🧪
*   **Mentalidad:** Supervivencia y Rentabilidad.
*   **Objetivo:** Probar el conocimiento adquirido en un entorno "congelado" (Out-of-Sample).
*   **Comportamiento del Oráculo:** Actúa como un **Sentinel**. Aplica filtros rigurosos y bloquea cualquier señal que no cumpla con los estándares de seguridad.
*   **Selectividad IA (>0%):** Aquí es donde la métrica de selectividad cobra vida, filtrando la "basura" que la IA aún no ha perfeccionado.
*   **PnL:** Debe ser positivo. Es la prueba real de que el sistema es apto para producción.

## 2. La Métrica de Selectividad IA 🛡️
La selectividad no es una métrica de "pérdida", sino de **inteligencia defensiva**. 
*   Un **0%** en Estudio no es un fallo; es una decisión de diseño para permitir la recolección de patrones.
*   El éxito se mide por la capacidad del sistema de pasar de un 0% (Estudio) a una rentabilidad consistente (Validación), gracias al filtro del Oráculo.

## 3. Entrenamiento Continuo 🌀
Cada ciclo utiliza meses diferentes. Esto evita el **Overfitting** (sobre-ajuste), forzando a la IA a adaptarse a mercados alcistas, bajistas y laterales de forma autónoma.

---
*Documento de Referencia Estratégica - Hapi Project*
