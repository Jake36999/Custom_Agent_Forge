from pydantic import BaseModel
import hashlib
import json

class BaseAletheiaModel(BaseModel):
    model_config = {"extra": "ignore", "populate_by_name": True, "strict": False}

    def stable_hash(self) -> str:
        # Deterministic hash of the model's data for integrity checks
        data = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
