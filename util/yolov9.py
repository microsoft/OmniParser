from contextlib import nullcontext
from pathlib import Path
from typing import Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.ops import batched_nms


DEFAULT_REPO_ID = "microsoft/OmniParser-v2.0"
DEFAULT_MODEL_FILE = "icon_detect_v3/model.pt"


class Boxes:
    def __init__(self, xyxy: torch.Tensor, confidence: torch.Tensor):
        self.xyxy = xyxy
        self.conf = confidence


class Result:
    def __init__(self, boxes: Boxes):
        self.boxes = boxes


class YOLOv9Detector:
    """TorchScript YOLOv9-E detector with an Ultralytics-compatible predict API."""

    strides = (8, 16, 32)

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        device: Union[str, torch.device, None] = None,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = "main",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {self.device}")

        if model_path is None:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=DEFAULT_MODEL_FILE,
                revision=revision,
            )
        self.model_path = Path(model_path)
        self.model = torch.jit.load(str(self.model_path), map_location=self.device).eval()

    @staticmethod
    def _normalize_image_size(image_size):
        if isinstance(image_size, int):
            width = height = image_size
        elif len(image_size) == 2:
            height, width = image_size
        else:
            raise ValueError(f"Expected one or two image dimensions, got {image_size}")
        width = ((int(width) + 31) // 32) * 32
        height = ((int(height) + 31) // 32) * 32
        return width, height

    @staticmethod
    def _load_image(source):
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source).convert("RGB")
        with Image.open(source) as image:
            return image.convert("RGB")

    def _preprocess(self, image, image_size):
        target_width, target_height = self._normalize_image_size(image_size)
        image_width, image_height = image.size
        scale = min(target_width / image_width, target_height / image_height)
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)
        pad_left = (target_width - resized_width) // 2
        pad_top = (target_height - resized_height) // 2

        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        padded = Image.new("RGB", (target_width, target_height), (114, 114, 114))
        padded.paste(resized, (pad_left, pad_top))
        image_array = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)
        return image_tensor, scale, pad_left, pad_top

    def _decode(self, outputs):
        class_logits = []
        decoded_boxes = []
        for output_index, stride in zip(range(0, len(outputs), 2), self.strides):
            layer_logits, layer_distances = outputs[output_index : output_index + 2]
            batch_size, _, height, width = layer_logits.shape
            layer_logits = layer_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, layer_logits.shape[1])
            layer_distances = layer_distances.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) * stride

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device),
                indexing="ij",
            )
            anchors = (torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2) + 0.5) * stride
            left_top, right_bottom = layer_distances.chunk(2, dim=-1)
            decoded_boxes.append(torch.cat((anchors - left_top, anchors + right_bottom), dim=-1))
            class_logits.append(layer_logits)

        return torch.cat(class_logits, dim=1).sigmoid(), torch.cat(decoded_boxes, dim=1)

    def _inference_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.inference_mode()
    def predict(self, source, conf=0.25, imgsz=640, iou=0.7, max_det=300):
        image = self._load_image(source)
        image_tensor, scale, pad_left, pad_top = self._preprocess(image, imgsz)

        with self._inference_context(), torch.jit.optimized_execution(False):
            class_scores, boxes = self._decode(self.model(image_tensor))

        scores, class_ids = class_scores[0].max(dim=-1)
        valid = scores > conf
        scores = scores[valid]
        class_ids = class_ids[valid]
        boxes = boxes[0][valid]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

        keep = batched_nms(boxes, scores, class_ids, iou)[:max_det]
        boxes = boxes[keep]
        scores = scores[keep]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image.height)
        return [Result(Boxes(boxes, scores))]
