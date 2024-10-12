from typing import Optional
from pydantic import BaseModel
from typing import Optional

class DataModel(BaseModel):
    Textos_espanol: str
    sdg: Optional[int] = None

def columns():
    return [
        "Textos_espanol",
        "sdg"
    ]

