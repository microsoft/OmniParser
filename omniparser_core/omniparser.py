import base64
import io
from typing import Dict

import torch
from PIL import Image

from omniparser_core.models import get_caption_model, get_yolo_model
from omniparser_core.pipeline import ScreenParserPipelineConfig, ScreenParserPipeline


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.yolo_model = get_yolo_model(model_path=config["som_model_path"])
        self.caption_model = get_caption_model(
            model_name=config["caption_model_name"],
            model_name_or_path=config["caption_model_path"],
            device=device,
        )
        self.pipeline = ScreenParserPipeline(
            yolo_model=self.yolo_model,
            caption_model=self.caption_model,
            config=ScreenParserPipelineConfig(box_threshold=config["BOX_TRESHOLD"]),
        )
        print("Omniparser initialized!!!")

    def parse(self, image_base64: str):
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print("image size:", image.size)
        dino_labled_img, label_coordinates, parsed_content_list = self.pipeline.parse_image(
            image
        )
        return dino_labled_img, parsed_content_list
