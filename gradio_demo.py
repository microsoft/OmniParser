from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import base64, os
import tempfile
from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""


def create_process_fn(yolo_model, caption_model_processor):
    def process(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz):
        # Convert image input to PIL Image if needed
        try:
            if isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif isinstance(image_input, dict) and "image" in image_input:
                # Handle image data from Gradio's image component
                image_data = image_input["image"]
                if isinstance(image_data, str) and image_data.startswith("data:image"):
                    # Handle base64 image string
                    image_data = image_data.split(",")[1]
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                elif isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data)
                else:
                    raise ValueError("Unsupported image data format")
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Unsupported image input format")

            if not isinstance(image, Image.Image):
                raise ValueError("Failed to convert input to PIL Image")

            box_overlay_ratio = image.size[0] / 3200
            draw_bbox_config = {
                "text_scale": 0.8 * box_overlay_ratio,
                "text_thickness": max(int(2 * box_overlay_ratio), 1),
                "text_padding": max(int(3 * box_overlay_ratio), 1),
                "thickness": max(int(3 * box_overlay_ratio), 1),
            }

            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image,
                display_img=False,
                output_bb_format="xyxy",
                goal_filtering=None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
                use_paddleocr=use_paddleocr,
            )
            text, ocr_bbox = ocr_bbox_rslt

            dino_labled_img, label_coordinates, parsed_content_list = (
                get_som_labeled_img(
                    image,
                    yolo_model,
                    BOX_TRESHOLD=box_threshold,
                    output_coord_in_ratio=True,
                    ocr_bbox=ocr_bbox,
                    draw_bbox_config=draw_bbox_config,
                    caption_model_processor=caption_model_processor,
                    ocr_text=text,
                    iou_threshold=iou_threshold,
                    imgsz=imgsz,
                )
            )

            output_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            print("finish processing")
            parsed_content_list = "\n".join(
                [f"icon {i}: " + str(v) for i, v in enumerate(parsed_content_list)]
            )
            return output_image, parsed_content_list
        except Exception as e:
            return None, str(e)

    return process


def create_gradio_demo(yolo_model=None, caption_model_processor=None):
    # Initialize models if not provided
    if yolo_model is None:
        yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
    if caption_model_processor is None:
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", model_name_or_path="weights/icon_caption_florence"
        )

    process_fn = create_process_fn(yolo_model, caption_model_processor)

    with gr.Blocks() as demo:
        gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                image_input_component = gr.Image(
                    type="pil",
                    label="Upload image",
                    sources=["upload", "clipboard"],
                    interactive=True,
                    height=400,
                )
                box_threshold_component = gr.Slider(
                    label="Box Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.05,
                )
                iou_threshold_component = gr.Slider(
                    label="IOU Threshold",
                    minimum=0.01,
                    maximum=1.0,
                    step=0.01,
                    value=0.1,
                )
                use_paddleocr_component = gr.Checkbox(label="Use PaddleOCR", value=True)
                imgsz_component = gr.Slider(
                    label="Icon Detect Image Size",
                    minimum=640,
                    maximum=1920,
                    step=32,
                    value=640,
                )
                submit_button_component = gr.Button(value="Submit", variant="primary")
            with gr.Column():
                image_output_component = gr.Image(type="pil", label="Image Output")
                text_output_component = gr.Textbox(
                    label="Parsed screen elements", placeholder="Text Output"
                )

        submit_button_component.click(
            fn=process_fn,
            inputs=[
                image_input_component,
                box_threshold_component,
                iou_threshold_component,
                use_paddleocr_component,
                imgsz_component,
            ],
            outputs=[image_output_component, text_output_component],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_demo()
    demo.launch(share=True, server_port=7861, server_name="0.0.0.0")
