# Rediseño de la Sección "Resultados y ML"

El objetivo es optimizar el espacio en la pestaña de Resultados, eliminando textos redundantes y reorganizando la información crítica (IA y Finanzas) en un diseño más profesional y compacto ("Cockpit" style).

## Cambios Propuestos

### Dashboard Frontend

#### [MODIFY] [index.html](file:///Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/dashboard/static/simulation/index.html)
-   **Eliminación de Texto Redundante**: Remover el párrafo explicativo sobre el pipeline de Machine Learning.
-   **Nueva Barra de Inteligencia (Intelligence Bar)**:
    -   Combinar el `neural-stats-card` y el `ai-advisor-detail-card` en una sola fila horizontal.
    -   Remover títulos largos como "Red Neuronal Adaptativa (MLP)".
    -   Usar iconos y etiquetas minimalistas para "Trades Aprendidos", "Accuracy" y "Recomendación IA".
-   **Compactación de Estadísticas Financieras**:
    -   Refinar el grid de rendimiento global para que los números sean el centro de atención.
    -   Usar fuentes monoespaciadas para los valores monetarios.

## Plan de Verificación

### Pruebas Visuales
-   Utilizar el `browser_subagent` para capturar una captura de pantalla del nuevo diseño.
-   Verificar que toda la información dinámica (muestras, precisión, pnl) se siga actualizando correctamente mediante WebSockets.

### Pruebas de Interacción
-   Confirmar que el diseño sea responsivo y se vea bien en diferentes anchos de pantalla.
