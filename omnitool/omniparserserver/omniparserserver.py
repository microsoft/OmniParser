import sys
import os
import time

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional

from util.omniparser import Omniparser


root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)


router = APIRouter()


class Config(BaseModel):
    some_model_path: Optional[str] = Field(default="weights/icon_detect/model.pt")
    caption_model_name: Optional[str] = Field(default="florence2")
    caption_model_path: Optional[str] = Field(default="weights/icon_caption_florence")
    device: Optional[str] = Field(default="cpu")


@router.post("/parse/")
async def parse(image: UploadFile = File(...), box_threshold: float = 0.05):
    print("start parsing...")
    config = Config()
    omniparser = Omniparser(
        config={
            "som_model_path": config.some_model_path,
            "caption_model_name": config.caption_model_name,
            "caption_model_path": config.caption_model_path,
            "device": config.device,
            "BOX_TRESHOLD": box_threshold,
        }
    )
    start = time.time()
    image_bytes = await image.read()
    dino_labled_img, parsed_content_list = omniparser.parse(image_bytes)
    latency = time.time() - start
    print("time:", latency)
    return {
        "image": dino_labled_img,
        "parsed_content_list": parsed_content_list,
        "latency": latency,
    }


@router.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}
