import docker
import os
import json
from pathlib import Path

def nuclear_wipe_execution():
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
        "completed_simulations.json",
        "active_sessions.json",
        "sim_history.json",
        "checkpoint.db",
        "state_sim.json",
        "state.json",
        "ml_dataset.csv",
        "training_history.csv"
    ]
    
    for filename in files_to_wipe:
        p = data_dir / filename
        if p.exists():
            print(f"🗑️ Eliminando: {filename}")
            p.unlink()

    # 3. Limpiar Directorios (Brain & Cache)
    dirs_to_clean = ["neural_models", "brain_backups", "cache"]
    for dname in dirs_to_clean:
        dpath = data_dir / dname
        if dpath.exists():
            print(f"🧹 Vaciando directorio: {dname}")
            for f in dpath.glob("*"):
                if f.is_file():
                    f.unlink()

    print("\n✨ NUCLEAR WIPE COMPLETADO. Todo está en CERO.")

if __name__ == "__main__":
    nuclear_wipe_execution()
