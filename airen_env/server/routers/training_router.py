from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/training", tags=["training"])

@router.get("/tis_status")
def get_tis_status() -> Dict[str, Any]:
    """Returns Truncated Importance Sampling (TIS) & Async GRPO active status."""
    return {
        "status": "active",
        "active_workers": 2,
        "queue_depth": 32,
        "buffer_mode": "multiprocessing"
    }
