from pydantic import BaseModel
from typing import List, Optional


class TaggingInRequest(BaseModel):
    image_path: str
    

class TaggingInResponse(BaseModel):
    status_code: int
    message: str
    infos: Optional[dict] = {}