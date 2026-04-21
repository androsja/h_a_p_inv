import docker
import os
import json
from pathlib import Path

def nuclear_wipe_execution(keep_history: bool = True):
    print("🚀 Iniciando LIMPIEZA TOTAL del Orquestador...")
    
    # 1. Detener y eliminar contenedores Docker
    print("📦 Limpiando contenedores Docker...")
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        count = 0
        for container in containers:
            if "hapi_worker_" in container.name:
                print(f"🛑 Matando contenedor: {container.name}...")
                container.stop(timeout=1)
                container.remove(force=True)
                count += 1
        print(f"✅ Se eliminaron {count} bots activos.")
    except Exception as e:
        print(f"❌ Error limpiando Docker: {e}")

    # 2. Limpiar Archivos de Datos (Nuclear)
    data_dir = Path("/Users/jflorezgaleano/Documents/JulianFlorez/Hapi/trading_bot/shared/data")
    print(f"📂 Limpiando archivos en {data_dir}...")
    
    files_to_wipe = [
        "trade_journal.csv",
        "backtest_results.json",
        "active_sessions.json",
        "sim_history.json",
        "checkpoint.db",
        "state_sim.json",
        "state.json",
        "ml_dataset.csv",
        "training_history.csv"
    ]

    # Archivos que se preservan a menos que sea un "Full Wipe"
    history_files = [
        "completed_simulations.json",
        "mastery_hub.json"
    ]
    
    if not keep_history:
        print("⚠️ MODO DESTRUCCIÓN TOTAL: Incluyendo historiales y maestría...")
        files_to_wipe.extend(history_files)
    
    for filename in files_to_wipe:
        p = data_dir / filename
        if p.exists():
            print(f"🗑️ Eliminando: {filename}")
            p.unlink()

    # 3. Limpiar Directorios (Brain & Cache) - Siempre se limpian para forzar re-entrenamiento si es necesario
    # Los modelos NEURALES solo se borran en Full Wipe para proteger el cerebro de los Elite
    dirs_to_clean = ["brain_backups", "cache"]
    if not keep_history:
        dirs_to_clean.append("neural_models")

    for dname in dirs_to_clean:
        dpath = data_dir / dname
        if dpath.exists():
            print(f"🧹 Vaciando directorio: {dname}")
            for f in dpath.glob("*"):
                if f.is_file():
                    f.unlink()

    msg = "✨ LIMPIEZA COMPLETADA (Historial Protegido)" if keep_history else "🔥 NUCLEAR WIPE COMPLETADO. Borrado absoluto."
    print(f"\n{msg}")

if __name__ == "__main__":
    import sys
    # Si se pasa "full" como argumento, se borra todo
    full_wipe = len(sys.argv) > 1 and sys.argv[1].lower() == "full"
    nuclear_wipe_execution(keep_history=not full_wipe)
