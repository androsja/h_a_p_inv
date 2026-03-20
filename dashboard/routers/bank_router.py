"""
routers/bank_router.py — Gestión del banco y fondos simulados.

Responsabilidades:
- Depósitos y retiros de la cuenta simulada
- WIPE total del sistema (borrar todos los datos y modelos)
"""

import json

from fastapi import APIRouter
from pydantic import BaseModel

from shared import config
from shared.config import (
    COMMAND_FILE, RESULTS_FILE, ML_DATASET_FILE, NEURAL_MODEL_FILE,
    LOG_FILE, CHECKPOINT_DB, STATE_FILE
)

router = APIRouter(prefix="/api", tags=["bank"])


# ─── Request Models ──────────────────────────────────────────────────────────

class DepositRequest(BaseModel):
    amount: float


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/bank/deposit")
async def bank_deposit(req: DepositRequest):
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
            data["available_cash"] = data.get("available_cash", config.INITIAL_CASH_LIVE) + req.amount
            with open(STATE_FILE, "w") as f:
                json.dump(data, f)
            return {"status": "success", "message": f"Deposited ${req.amount:.2f}", "new_balance": data["available_cash"]}
        return {"status": "error", "message": "State file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/bank/withdraw")
async def bank_withdraw(req: DepositRequest):
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
            withdrawal_fee = 4.99
            total_deduction = req.amount + withdrawal_fee
            if data.get("available_cash", config.INITIAL_CASH_LIVE) >= total_deduction:
                data["available_cash"] -= total_deduction
                with open(STATE_FILE, "w") as f:
                    json.dump(data, f)
                return {"status": "success", "message": f"Withdrew ${req.amount:.2f} (Fee: ${withdrawal_fee:.2f})", "new_balance": data["available_cash"]}
            else:
                return {"status": "error", "message": "Insufficient funds including $4.99 withdrawal fee."}
        return {"status": "error", "message": "State file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/wipe_total")
async def wipe_total():
    """Borra ABSOLUTAMENTE TODO y reinicia el sistema desde cero."""
    import shutil
    deleted = []
    errors = []

    try:
        cmd_data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                cmd_data = json.load(f)
        cmd_data.update({
            "live_stop": True, "live_start": False,
            "reset_all": True, "reset_neural": True,
            "strategy_frozen": False
        })
        with open(COMMAND_FILE, "w") as f:
            json.dump(cmd_data, f)
    except Exception as e:
        errors.append(f"config_reset: {e}")

    files_to_delete = [
        # Reiniciar logs y modelos
        ("neural_model.joblib",   NEURAL_MODEL_FILE),
        ("ml_dataset.csv",        ML_DATASET_FILE),
        ("trade_journal.csv",     config.TRADE_JOURNAL_FILE),
        ("state_live.json",       config.STATE_FILE_LIVE),
        ("state_sim.json",        config.STATE_FILE_SIM),
        ("backtest_results.json", config.RESULTS_FILE),
        ("checkpoint.db",         CHECKPOINT_DB),
        ("training_history.csv",  config.TRAINING_LOG_FILE),
        ("sim_history.json",      config.DATA_CACHE_DIR / "sim_history.json"),
    ]

    for name, path in files_to_delete:
        try:
            if path.exists():
                path.unlink()
                deleted.append(name)
        except Exception as e:
            errors.append(f"{name}: {e}")

    # Borrar snapshots guardados
    try:
        if config.MODEL_SNAPSHOTS_DIR.exists():
            shutil.rmtree(config.MODEL_SNAPSHOTS_DIR)
            config.MODEL_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            deleted.append("snapshots_dir")
    except Exception as e:
        errors.append(f"snapshots_dir: {e}")

    try:
        if LOG_FILE.exists(): LOG_FILE.write_text("")
    except: pass

    verify_ok = not ML_DATASET_FILE.exists()
    return {
        "status": "success" if not errors else "partial",
        "deleted": deleted,
        "errors": errors,
        "verify": verify_ok,
        "message": f"✅ WIPE TOTAL completado. {len(deleted)} componentes eliminados (incluyendo historial y snapshots)."
    }


@router.post("/wipe_neural")
async def wipe_neural_alias():
    return await wipe_total()
