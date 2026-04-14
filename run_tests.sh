#!/bin/bash
echo "🚀 HAPI BOTS: Ejecutando Suite de Pruebas Automatizadas Rápidas..."
python3 -m pytest tests/ -v
if [ $? -eq 0 ]; then
    echo "✅ Todas las pruebas pasaron exitosamente. Seguro para hacer commit."
else
    echo "❌ Algunas pruebas fallaron. Revisa los errores arriba."
fi
