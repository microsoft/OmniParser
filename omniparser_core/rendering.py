import numpy as np
import supervision as sv
import torch
from torchvision.ops import box_convert

from omniparser_core.box_annotator import BoxAnnotator


def get_xywh(input_box):
    x = input_box[0][0]
    y = input_box[0][1]
    w = input_box[2][0] - input_box[0][0]
    h = input_box[2][1] - input_box[0][1]
    return int(x), int(y), int(w), int(h)


def get_xyxy(input_box):
    x = input_box[0][0]
    y = input_box[0][1]
    xp = input_box[2][0]
    yp = input_box[2][1]
    return int(x), int(y), int(xp), int(yp)


def get_xywh_yolo(input_box):
    x = input_box[0]
    y = input_box[1]
    w = input_box[2] - input_box[0]
    h = input_box[3] - input_box[1]
    return int(x), int(y), int(w), int(h)


def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases,
    text_scale: float,
    text_padding=5,
    text_thickness=2,
    thickness=3,
):
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)
    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]
    box_annotator = BoxAnnotator(
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=text_thickness,
        thickness=thickness,
    )
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h)
    )
    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates
