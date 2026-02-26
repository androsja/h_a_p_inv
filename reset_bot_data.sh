#!/bin/bash

# Este script borra todos los archivos de "memoria" del bot en simulación de manera robusta.
# Útil cuando deseas arrancar la bitácora desde 0% y reiniciar el entrenamiento ML.

echo "🧹 Deteniendo el bot y el dashboard temporalmente..."
docker stop hapi_bot_simulated hapi_dashboard >/dev/null 2>&1

echo "🧹 Borrando bases de datos, bitácoras y memoria compartida del volumen..."
# Borramos sobre el volumen exacto que levantó docker-compose
docker run --rm -v trading_bot_bot_data:/app/data alpine rm -f /app/data/backtest_results.json /app/data/ml_dataset.csv /app/data/state.json /app/data/command.json

echo "🔄 Reiniciando el simulador (Trading Bot & Dashboard) para aplicar los cambios y comenzar..."
docker start hapi_bot_simulated hapi_dashboard >/dev/null 2>&1

echo "✅ ¡Listo! El sistema entero ha sido reiniciado con la memoria en blanco absoluta. El explorador comenzará desde el 0%."
