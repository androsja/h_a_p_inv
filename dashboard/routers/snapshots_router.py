"""
routers/snapshots_router.py — Versiones de IA e Historial de Simulaciones.

Responsabilidades:
- Guardar, listar, restaurar y eliminar versiones del modelo de IA
- Guardar y recuperar el historial de simulaciones completadas
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from shared.config import (
    NEURAL_MODEL_FILE, ML_DATASET_FILE, MODEL_SNAPSHOTS_DIR,
    COMMAND_FILE, DATA_CACHE_DIR
)

router = APIRouter(prefix="/api", tags=["snapshots"])


# ─── Request Models ──────────────────────────────────────────────────────────

class SnapshotRequest(BaseModel):
    label: str

class RestoreRequest(BaseModel):
    snapshot_id: str

class SimHistoryRequest(BaseModel):
    timestamp: str
    symbols_count: int
    trades_learned: int
    total_ghosts: int = 0  # 👻 Nuevo campo para auditoría
    win_rate: float
    accuracy: float
    pnl: float
    symbols_list: list[str] = []
    sim_start: str = "─"
    sim_end: str = "─"
    investment_style: str = "Normal"


# ─── Model Snapshot Endpoints ─────────────────────────────────────────────────

@router.post("/model_snapshot")
async def save_model_snapshot(req: SnapshotRequest):
    """Guarda una copia de los modelos de IA como versión nombrada."""
    import shutil
    try:
        MODEL_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_ " else "_" for c in req.label)[:40]
        snap_dir = MODEL_SNAPSHOTS_DIR / f"{ts}_{safe_label}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        files_saved = []
        if NEURAL_MODEL_FILE.exists():
            shutil.copy2(NEURAL_MODEL_FILE, snap_dir / "neural_model.joblib")
            files_saved.append("neural_model.joblib")
        if ML_DATASET_FILE.exists():
            shutil.copy2(ML_DATASET_FILE, snap_dir / "ml_dataset.csv")
            files_saved.append("ml_dataset.csv")

        if not files_saved:
            snap_dir.rmdir()
            return {"status": "error", "message": "No hay modelos entrenados para guardar. Corre al menos una simulación primero."}

        # NEW: Capturar los resultados detallados de la simulación actual
        res_file = Path(__file__).parent.parent.parent / "data" / "backtest_results.json"
        detailed_results = []
        if res_file.exists():
            try:
                with open(res_file, "r") as f:
                    detailed_results = json.load(f)
            except: pass

        meta = {
            "id": f"{ts}_{safe_label}",
            "label": req.label,
            "created_at": datetime.utcnow().isoformat(),
            "files": files_saved,
            "detailed_results": detailed_results # Persistimos el estado de los símbolos en este snapshot
        }
        with open(snap_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"status": "success", "message": f"✅ Versión '{req.label}' guardada", "snapshot_id": meta["id"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/model_snapshots")
async def list_model_snapshots():
    """Lista todas las versiones de IA guardadas."""
    try:
        MODEL_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshots = []
        for snap_dir in sorted(MODEL_SNAPSHOTS_DIR.iterdir(), reverse=True):
            if not snap_dir.is_dir(): continue
            meta_file = snap_dir / "meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                mlp_path = snap_dir / "neural_model.joblib"
                meta["mlp_size_kb"] = round(mlp_path.stat().st_size / 1024, 1) if mlp_path.exists() else 0
                snapshots.append(meta)
        return {"status": "success", "snapshots": snapshots}
    except Exception as e:
        return {"status": "error", "snapshots": [], "message": str(e)}


@router.post("/restore_snapshot")
async def restore_model_snapshot(req: RestoreRequest):
    """Restaura una versión guardada de la IA."""
    import shutil
    try:
        snap_dir = MODEL_SNAPSHOTS_DIR / req.snapshot_id
        if not snap_dir.exists():
            return {"status": "error", "message": f"Versión '{req.snapshot_id}' no encontrada."}

        restored = []
        mlp_snap = snap_dir / "neural_model.joblib"
        if mlp_snap.exists():
            shutil.copy2(mlp_snap, NEURAL_MODEL_FILE)
            restored.append("neural_model.joblib")

        csv_snap = snap_dir / "ml_dataset.csv"
        if csv_snap.exists():
            shutil.copy2(csv_snap, ML_DATASET_FILE)
            restored.append("ml_dataset.csv")

        if not restored:
            return {"status": "error", "message": "La versión seleccionada no tiene archivos de modelo."}

        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        data["reload_models"] = True
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)

        meta_file = snap_dir / "meta.json"
        label = req.snapshot_id
        if meta_file.exists():
            with open(meta_file) as f:
                label = json.load(f).get("label", req.snapshot_id)

        return {"status": "success", "message": f"✅ Versión '{label}' restaurada.", "restored": restored}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.delete("/model_snapshot/{snapshot_id}")
async def delete_model_snapshot(snapshot_id: str):
    """Elimina una versión guardada de la IA."""
    import shutil
    try:
        snap_dir = MODEL_SNAPSHOTS_DIR / snapshot_id
        if not snap_dir.exists():
            return {"status": "error", "message": "Versión no encontrada."}
        shutil.rmtree(snap_dir)
        return {"status": "success", "message": f"Versión '{snapshot_id}' eliminada."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── Simulation History Endpoints ─────────────────────────────────────────────

@router.post("/sim_history")
async def save_sim_history(req: SimHistoryRequest):
    try:
        history_file = DATA_CACHE_DIR / "sim_history.json"
        history = []
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
        # NEW: Adjuntar los resultados detallados por símbolo a este hito de la historia
        res_file = Path(__file__).parent.parent.parent / "data" / "backtest_results.json"
        detailed_results = []
        if res_file.exists():
            try:
                with open(res_file, "r") as f:
                    detailed_results = json.load(f)
            except: pass
        
        entry = req.dict()
        entry["detailed_results"] = detailed_results
        
        history.append(entry)
        history = history[-500:]  # Keep last 500 simulations
        with open(history_file, "w") as f:
            json.dump(history, f)
        return {"status": "success", "message": "Simulation saved to history"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/sim_history")
async def get_sim_history():
    try:
        history_file = DATA_CACHE_DIR / "sim_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)
            return {"status": "success", "history": history[::-1]}  # newest first
        return {"status": "success", "history": []}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/clear_sim_history")
async def clear_sim_history():
    """Borra únicamente el historial de simulaciones (sim_history.json)."""
    try:
        history_file = DATA_CACHE_DIR / "sim_history.json"
        if history_file.exists():
            history_file.unlink()
            return {"status": "success", "message": "✅ Historial de simulaciones borrado correctamente."}
        return {"status": "success", "message": "El historial ya estaba vacío."}
    except Exception as e:
        return {"status": "error", "message": f"Error al borrar el historial: {str(e)}"}
