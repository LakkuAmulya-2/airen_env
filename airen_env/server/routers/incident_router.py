from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/incidents", tags=["incidents"])

@router.get("/")
def get_incidents() -> Dict[str, Any]:
    """Returns the list of all active production incident scenarios."""
    return {
        "status": "success",
        "incident_types": [
            "db_overload", "memory_leak", "network_partition", 
            "bad_deployment", "cache_stampede", "api_timeout", 
            "disk_full", "ssl_cert_expired", "ddos_attack"
        ]
    }
