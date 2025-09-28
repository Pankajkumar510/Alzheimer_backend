from typing import Any, Dict, Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    status: str
    result: Dict[str, Any]


class CognitiveOnlyRequest(BaseModel):
    patient_id: str
    cognitive: Dict[str, Any]
