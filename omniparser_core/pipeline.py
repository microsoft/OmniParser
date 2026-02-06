from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

from omniparser_core.captioning import (
    get_parsed_content_icon,
    get_parsed_content_icon_phi3v,
)
from omniparser_core.detection import predict_yolo
from omniparser_core.ocr import check_ocr_box
from omniparser_core.postprocess import int_box_area, remove_overlap_new
from omniparser_core.rendering import annotate


@dataclass
class ScreenParserPipelineConfig:
    box_threshold: float = 0.05
    ocr_text_threshold: float = 0.8
    use_paddleocr: bool = False
    iou_threshold: float = 0.7
    batch_size: int = 128
    scale_img: bool = False
    output_coord_in_ratio: bool = True
    use_local_semantics: bool = True


@dataclass
class OCRResult:
    text: list[str]
    boxes_xyxy: list[list[int]]


@dataclass
class DetectionResult:
    icon_boxes_xyxy_ratio: torch.Tensor
    logits: torch.Tensor


def build_draw_bbox_config(image: Image.Image) -> dict[str, float | int]:
    box_overlay_ratio = max(image.size) / 3200
    return {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }


class ScreenParserPipeline:
    def __init__(
        self,
        yolo_model: Any,
        caption_model: Any,
        config: ScreenParserPipelineConfig | None = None,
    ) -> None:
        self.som_model = yolo_model
        self.caption_model_processor = caption_model
        self.config = config or ScreenParserPipelineConfig()

    def run_ocr(self, image: Image.Image) -> OCRResult:
        (text, ocr_bbox), _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            easyocr_args={"text_threshold": self.config.ocr_text_threshold},
            use_paddleocr=self.config.use_paddleocr,
        )
        return OCRResult(text=text, boxes_xyxy=ocr_bbox)

    def run_icon_detection(self, image: Image.Image) -> DetectionResult:
        width, height = image.size
        imgsz = (height, width)
        xyxy, logits, _ = predict_yolo(
            self.som_model,
            image=image,
            box_threshold=self.config.box_threshold,
            imgsz=imgsz,
            scale_img=self.config.scale_img,
            iou_threshold=0.1,
        )
        xyxy_ratio = xyxy / torch.Tensor([width, height, width, height]).to(xyxy.device)
        return DetectionResult(icon_boxes_xyxy_ratio=xyxy_ratio, logits=logits)

    def build_elements(
        self,
        ocr_result: OCRResult,
        detection_result: DetectionResult,
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], list[list[float]] | None]:
        if ocr_result.boxes_xyxy:
            ocr_bbox_ratio = (
                torch.tensor(ocr_result.boxes_xyxy)
                / torch.Tensor([width, height, width, height])
            ).tolist()
        else:
            ocr_bbox_ratio = None

        ocr_elements = [
            {
                "type": "text",
                "bbox": box,
                "interactivity": False,
                "content": txt,
                "source": "box_ocr_content_ocr",
            }
            for box, txt in zip(ocr_bbox_ratio or [], ocr_result.text)
            if int_box_area(box, width, height) > 0
        ]
        icon_elements = [
            {"type": "icon", "bbox": box, "interactivity": True, "content": None}
            for box in detection_result.icon_boxes_xyxy_ratio.tolist()
            if int_box_area(box, width, height) > 0
        ]
        filtered_elements = remove_overlap_new(
            boxes=icon_elements,
            iou_threshold=self.config.iou_threshold,
            ocr_bbox=ocr_elements,
        )
        filtered_elements = sorted(filtered_elements, key=lambda x: x["content"] is None)
        return filtered_elements, ocr_bbox_ratio

    def run_icon_caption(
        self,
        filtered_elements: list[dict[str, Any]],
        ocr_bbox_ratio: list[list[float]] | None,
        image_np: np.ndarray,
    ) -> None:
        if not self.config.use_local_semantics:
            return
        if not filtered_elements:
            return

        filtered_boxes = torch.tensor([box["bbox"] for box in filtered_elements])
        starting_idx = next(
            (i for i, box in enumerate(filtered_elements) if box["content"] is None), -1
        )
        if starting_idx < 0:
            return

        caption_model = self.caption_model_processor["model"]
        if "phi3_v" in caption_model.config.model_type:
            icon_captions = get_parsed_content_icon_phi3v(
                filtered_boxes,
                ocr_bbox_ratio,
                image_np,
                self.caption_model_processor,
            )
        else:
            icon_captions = get_parsed_content_icon(
                filtered_boxes,
                starting_idx,
                image_np,
                self.caption_model_processor,
                batch_size=self.config.batch_size,
            )

        for box in filtered_elements:
            if box["content"] is None and icon_captions:
                box["content"] = icon_captions.pop(0)

    def render_labeled_image(
        self,
        image_np: np.ndarray,
        filtered_elements: list[dict[str, Any]],
        logits: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[str, dict[str, list[float]]]:
        width, height = image_size
        if filtered_elements:
            filtered_boxes = torch.tensor([box["bbox"] for box in filtered_elements])
            filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
        else:
            filtered_boxes = torch.empty((0, 4))
        draw_cfg = build_draw_bbox_config(Image.fromarray(image_np))
        phrases = [i for i in range(len(filtered_boxes))]
        annotated_frame, label_coordinates = annotate(
            image_source=image_np,
            boxes=filtered_boxes,
            logits=logits,
            phrases=phrases,
            **draw_cfg,
        )
        buffered = io.BytesIO()
        Image.fromarray(annotated_frame).save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")

        if self.config.output_coord_in_ratio:
            label_coordinates = {
                k: [v[0] / width, v[1] / height, v[2] / width, v[3] / height]
                for k, v in label_coordinates.items()
            }
        return encoded_image, label_coordinates

    def parse_image(
        self, image: Image.Image
    ) -> tuple[str, dict[str, list[float]], list[dict[str, Any]]]:
        image = image.convert("RGB")
        width, height = image.size
        image_np = np.asarray(image)

        ocr_result = self.run_ocr(image)
        detection_result = self.run_icon_detection(image)
        filtered_elements, ocr_bbox_ratio = self.build_elements(
            ocr_result, detection_result, width, height
        )
        self.run_icon_caption(filtered_elements, ocr_bbox_ratio, image_np)
        encoded_image, label_coordinates = self.render_labeled_image(
            image_np, filtered_elements, detection_result.logits, (width, height)
        )
        return encoded_image, label_coordinates, filtered_elements
