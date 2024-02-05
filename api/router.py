from fastapi import APIRouter, HTTPException, status
from api.base_model import *

from modules.tagging import TaggingEngine
from utils.tools import base64_to_image, image_to_base64

tagging_router = APIRouter()

tagging_engine = TaggingEngine(model_path= 'Qwen/Qwen-VL-Chat-Int4')


# GET method for health check
@tagging_router.get("/api/tagging/state/health-check")
async def health_check():
    return {"message": "HangersAI ReColor API is running!"}


# POST method for tagging
@tagging_router.post("/api/tagging/inference/general-tagging", status_code=status.HTTP_200_OK)
def general_tagging(tagging_in_request: TaggingInRequest):
    # Inference here
    image_path = tagging_in_request.image_path
    prediction = tagging_engine.general_tagging(image_path=image_path)
    
    return TaggingInResponse(status_code=prediction["status_code"], 
                             message=prediction["message"], 
                             infos=prediction["infos"])


