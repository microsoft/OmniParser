from typing import Union

import cv2
import easyocr
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR

from omniparser_core.rendering import get_xywh, get_xyxy

reader = easyocr.Reader(["en"])


def _init_paddle_ocr():
    kwargs = {
        "lang": "en",
        "use_angle_cls": False,
        "use_gpu": True,
        "show_log": False,
        "max_batch_size": 1024,
        "use_dilation": True,
        "det_db_score_mode": "slow",
        "rec_batch_num": 1024,
    }
    while True:
        try:
            return PaddleOCR(**kwargs)
        except ValueError as e:
            msg = str(e)
            prefix = "Unknown argument: "
            if prefix not in msg:
                raise
            bad_key = msg.split(prefix, 1)[1].strip().split()[0].strip(",")
            if bad_key not in kwargs:
                raise
            kwargs.pop(bad_key, None)


paddle_ocr = _init_paddle_ocr()


def _run_paddle_ocr(image_np, text_threshold):
    try:
        try:
            result = paddle_ocr.ocr(image_np, cls=False)[0]
        except TypeError:
            result = paddle_ocr.ocr(image_np)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
        return text, coord
    except Exception as e:
        print(f"PaddleOCR failed ({type(e).__name__}): {e}. Falling back to EasyOCR.")
        result = reader.readtext(image_np)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
        return text, coord


def check_ocr_box(
    image_source: Union[str, Image.Image],
    display_img=True,
    output_bb_format="xywh",
    goal_filtering=None,
    easyocr_args=None,
    use_paddleocr=False,
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == "RGBA":
        image_source = image_source.convert("RGB")
    image_np = np.array(image_source)

    if use_paddleocr:
        text_threshold = 0.5 if easyocr_args is None else easyocr_args["text_threshold"]
        text, coord = _run_paddle_ocr(image_np, text_threshold)
    else:
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]

    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x + a, y + b), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == "xywh":
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == "xyxy":
            bb = [get_xyxy(item) for item in coord]
        else:
            raise ValueError(f"Unsupported output_bb_format: {output_bb_format}")
    return (text, bb), goal_filtering
