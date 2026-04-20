import docker
import os
import json
import requests
from pathlib import Path

def truly_nuclear_cold_reset():
    print("❄️ Iniciando RESET FRÍO (Deteniendo Motores)...")
    
    # 1. Detener el AutoTrainer (esto evita que relance bots mientras limpiamos)
    try:
        resp = requests.post("http://localhost:9000/api/autotrainer/toggle", json={"action": "stop"}, timeout=5)
        print(f"🛑 Autotrainer detenido: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ No se pudo detener el Autotrainer por API (posiblemente ya parado o inaccesible): {e}")

    # 2. Matar y Limpiar Docker
    print("📦 Limpiando contenedores Docker...")
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        count = 0
        for container in containers:
            if "hapi_worker_" in container.name:
                print(f"🛑 Matando contenedor: {container.name}...")
                container.remove(force=True)
                count += 1
        print(f"✅ Se eliminaron {count} bots.")
    except Exception as e:
        print(f"❌ Error limpiando Docker: {e}")

    # 3. Limpiar Archivos de Datos (Nuclear)
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
        "training_history.csv",
        "mastery_hub.json"
    ]
    
    for filename in files_to_wipe:
        p = data_dir / filename
        if p.exists():
            print(f"🗑️ Eliminando: {filename}")
            p.unlink()

    # 4. Limpiar Directorios
    dirs_to_clean = ["neural_models", "brain_backups", "cache"]
    for dname in dirs_to_clean:
        dpath = data_dir / dname
        if dpath.exists():
            print(f"🧹 Vaciando directorio: {dname}")
            for f in dpath.glob("*"):
                if f.is_file():
                    f.unlink()

    print("\n✨ RESET FRÍO COMPLETADO. El sistema está 100% CERO.")

if __name__ == "__main__":
    truly_nuclear_cold_reset()
