import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
from openai import AzureOpenAI
import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
from paddleocr import PaddleOCR
from loguru import logger
import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from util.box_annotator import BoxAnnotator

# Initialize EasyOCR and PaddleOCR
reader = easyocr.Reader(['en'])
paddle_ocr = PaddleOCR(
    lang='en',
    use_angle_cls=False,
    use_gpu=True,
    show_log=False,
    max_batch_size=128,
    use_dilation=True,
    det_db_score_mode='slow',
    rec_batch_num=128,
    device="cuda"
)

# paddle_ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False)

def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    logger.info(f"Initializing caption model processor: {model_name}")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Selected device: {device}")
   
    try:
        if model_name == "blip2":
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            logger.debug("Loading Blip2 processor and model")
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                device_map=None,
                torch_dtype=torch.float32 if device == 'cpu' else torch.float16
            ).to(device)
        elif model_name == "florence2":
            from transformers import AutoProcessor, AutoModelForCausalLM
            logger.debug("Loading Florence2 processor and model")
            processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32 if device == 'cpu' else torch.float16,
                trust_remote_code=True
            ).to(device)
        else:
            logger.error(f"Unsupported model name: {model_name}")
            raise ValueError(f"Unsupported model: {model_name}")
       
        logger.success(f"Successfully loaded {model_name} model and processor")
        return {'model': model.to(device), 'processor': processor}
    except Exception as e:
        logger.error(f"Failed to load caption model processor: {str(e)}")
        raise

def get_yolo_model(model_path):
    from ultralytics import YOLO
    logger.info(f"Loading YOLO model from: {model_path}")
    try:
        model = YOLO(model_path)
        logger.success("Successfully loaded YOLO model")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {str(e)}")
        raise

@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
    logger.info(f"Starting icon content parsing with {len(filtered_boxes)} boxes, batch_size={batch_size}")
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
   
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
            # logger.debug(f"Processed crop for box {i}: ({xmin},{ymin},{xmax},{ymax})")
        except Exception as e:
            logger.error(f"Error cropping box {i}: {str(e)}")
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        prompt = "<CAPTION>" if 'florence' in model.config.name_or_path else "The image shows"
    logger.debug(f"Using prompt: {prompt}")
   
    generated_texts = []
    device = model.device
   
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i:i+batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} images")
       
        try:
            inputs = processor(
                images=batch,
                text=[prompt]*len(batch),
                return_tensors="pt",
                do_resize=False
            ).to(device=device, dtype=torch.float16 if model.device.type == 'cuda' else torch.float32)
           
            if 'florence' in model.config.name_or_path:
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20,
                    num_beams=1,
                    do_sample=False
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    num_return_sequences=1
                )
               
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = [gen.strip() for gen in generated_text]
            generated_texts.extend(generated_text)
            logger.debug(f"Batch processing time: {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue
   
    logger.success(f"Completed icon content parsing, generated {len(generated_texts)} captions")
    return generated_texts

def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    logger.info(f"Starting phi3v icon content parsing with {len(filtered_boxes)} boxes")
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
   
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            croped_pil_image.append(to_pil(cropped_image))
            logger.debug(f"Processed crop for box {i}: ({xmin},{ymin},{xmax},{ymax})")
        except Exception as e:
            logger.error(f"Error cropping box {i}: {str(e)}")
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   
    batch_size = 5
    generated_texts = []
   
    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1} with {len(images)} images")
       
        try:
            image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
            inputs = {'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
            texts = [prompt] * len(images)
           
            for j, txt in enumerate(texts):
                input = processor._convert_images_texts_to_inputs(image_inputs[j], txt, return_tensors="pt")
                inputs['input_ids'].append(input['input_ids'])
                inputs['attention_mask'].append(input['attention_mask'])
                inputs['pixel_values'].append(input['pixel_values'])
                inputs['image_sizes'].append(input['image_sizes'])
           
            max_len = max([x.shape[1] for x in inputs['input_ids']])
            for j, v in enumerate(inputs['input_ids']):
                inputs['input_ids'][j] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
                inputs['attention_mask'][j] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][j]], dim=1)
            inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}
           
            generation_args = {
                "max_new_tokens": 25,
                "temperature": 0.01,
                "do_sample": False,
            }
            generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
            generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            response = [res.strip('\n').strip() for res in response]
            generated_texts.extend(response)
            logger.debug(f"Processed batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue
   
    logger.success(f"Completed phi3v icon content parsing, generated {len(generated_texts)} captions")
    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    # logger.info(f"Removing overlapping boxes with IoU threshold: {iou_threshold}")
    assert ocr_bbox is None or isinstance(ocr_bbox, List)
   
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
   
    for i, box1 in enumerate(boxes):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                # logger.debug(f"Removed box {i} due to overlap with box {j}")
                break
        if is_valid_box:
            if ocr_bbox:
                if not any(IoU(box1, box3) > iou_threshold and not is_inside(box1, box3) for box3 in ocr_bbox):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
   
    # logger.success(f"Filtered boxes, kept {len(filtered_boxes)} out of {len(boxes)}")
    return torch.tensor(filtered_boxes)

def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    # logger.info(f"Removing overlapping boxes (new) with IoU threshold: {iou_threshold}")
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
   
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                # logger.debug(f"Removed box {i} due to overlap with box {j}")
                break
        if is_valid_box:
            if ocr_bbox:
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1):
                            try:
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                                # logger.debug(f"Removed OCR box inside icon box {i}")
                            except:
                                continue
                        elif is_inside(box1, box3):
                            box_added = True
                            # logger.debug(f"Skipped icon box {i} inside OCR box")
                            break
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({
                            'type': 'icon',
                            'bbox': box1_elem['bbox'],
                            'interactivity': True,
                            'content': ocr_labels,
                            'source': 'box_yolo_content_ocr'
                        })
                    else:
                        filtered_boxes.append({
                            'type': 'icon',
                            'bbox': box1_elem['bbox'],
                            'interactivity': True,
                            'content': None,
                            'source': 'box_yolo_content_yolo'
                        })
            else:
                filtered_boxes.append(box1)
   
    # logger.success(f"Filtered boxes (new), kept {len(filtered_boxes)} out of {len(boxes)}")
    return filtered_boxes

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    logger.info(f"Loading image from: {image_path}")
    try:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        logger.success(f"Successfully loaded and transformed image: {image_path}")
        return image, image_transformed
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        raise

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float,
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    logger.info(f"Annotating image with {len(boxes)} boxes")
    try:
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
            thickness=thickness
        )
        annotated_frame = image_source.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
            image_size=(w,h)
        )
        label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
        logger.success("Image annotation completed")
        return annotated_frame, label_coordinates
    except Exception as e:
        logger.error(f"Error during image annotation: {str(e)}")
        raise

def predict(model, image, caption, box_threshold, text_threshold):
    logger.info(f"Running prediction with box_threshold={box_threshold}, text_threshold={text_threshold}")
    try:
        device = model.device
        inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
       
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
        logger.success(f"Prediction completed, found {len(boxes)} objects")
        return boxes, logits, phrases
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    logger.info(f"Running YOLO prediction with box_threshold={box_threshold}, iou_threshold={iou_threshold}")
    try:
        if scale_img:
            result = model.predict(
                source=image,
                conf=box_threshold,
                imgsz=imgsz,
                iou=iou_threshold,
            )
        else:
            result = model.predict(
                source=image,
                conf=box_threshold,
                iou=iou_threshold,
            )
        boxes = result[0].boxes.xyxy
        conf = result[0].boxes.conf
        phrases = [str(i) for i in range(len(boxes))]
        logger.success(f"YOLO prediction completed, found {len(boxes)} objects")
        return boxes, conf, phrases
    except Exception as e:
        logger.error(f"YOLO prediction failed: {str(e)}")
        raise

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    # logger.debug(f"Calculated box area: {area}")
    return area

def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False,
                       ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None,
                       caption_model_processor=None, ocr_text=[], use_local_semantics=True,
                       iou_threshold=0.9, prompt=None, scale_img=False, imgsz=None, batch_size=128):
    logger.info(f"Processing image for SOM labeling, use_local_semantics={use_local_semantics}")
    try:
        if isinstance(image_source, str):
            image_source = Image.open(image_source)
        image_source = image_source.convert("RGB")
        w, h = image_source.size
        if not imgsz:
            imgsz = (h, w)
        logger.debug(f"Image size: {w}x{h}")
       
        xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD,
                                           imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
        xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
        image_source = np.asarray(image_source)
        phrases = [str(i) for i in range(len(phrases))]
       
        if ocr_bbox:
            ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
            ocr_bbox = ocr_bbox.tolist()
        else:
            logger.debug("No OCR bounding boxes provided")
            ocr_bbox = None
       
        ocr_bbox_elem = [
            {'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt, 'source': 'box_ocr_content_ocr'}
            for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0
        ]
        xyxy_elem = [
            {'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None}
            for box in xyxy.tolist() if int_box_area(box, w, h) > 0
        ]
        filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
       
        filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
        starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
        filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
        logger.debug(f"Filtered boxes: {len(filtered_boxes)}, starting_idx: {starting_idx}")
       
        time1 = time.time()
        if use_local_semantics:
            caption_model = caption_model_processor['model']
            if 'phi3_v' in caption_model.config.model_type:
                parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
            else:
                parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source,
                                                            caption_model_processor, prompt=prompt, batch_size=batch_size)
            ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
            icon_start = len(ocr_text)
            parsed_content_icon_ls = [f"Icon Box ID {str(i+icon_start)}: {txt}" for i, txt in enumerate(parsed_content_icon)]
           
            for i, box in enumerate(filtered_boxes_elem):
                if box['content'] is None:
                    box['content'] = parsed_content_icon.pop(0)
            parsed_content_merged = ocr_text + parsed_content_icon_ls
        else:
            parsed_content_merged = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        logger.debug(f"Time to get parsed content: {time.time()-time1:.2f}s")
       
        filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
        phrases = [i for i in range(len(filtered_boxes))]
       
        if draw_bbox_config:
            annotated_frame, label_coordinates = annotate(
                image_source=image_source,
                boxes=filtered_boxes,
                logits=logits,
                phrases=phrases,
                **draw_bbox_config
            )
        else:
            annotated_frame, label_coordinates = annotate(
                image_source=image_source,
                boxes=filtered_boxes,
                logits=logits,
                phrases=phrases,
                text_scale=text_scale,
                text_padding=text_padding
            )
       
        pil_img = Image.fromarray(annotated_frame)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
        if output_coord_in_ratio:
            label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
       
        logger.success("SOM labeling completed successfully")
        return encoded_image, label_coordinates, filtered_boxes_elem
    except Exception as e:
        logger.error(f"Error in SOM labeling: {str(e)}")
        raise

# def get_xywh(input):
#     x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
#     x, y, w, h = int(x), int(y), int(w), int(h)
#     logger.debug(f"Converted to xywh: ({x},{y},{w},{h})")
#     return x, y, w, h

# def get_xyxy(input):
#     x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
#     x, y, xp, yp = int(x), int(y), int(xp), int(yp)
#     logger.debug(f"Converted to xyxy: ({x},{y},{xp},{yp})")
#     return x, y, xp, yp

# def get_xywh_yolo(input):
#     x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
#     x, y, w, h = int(x), int(y), int(w), int(h)
#     logger.debug(f"Converted YOLO to xywh: ({x},{y},{w},{h})")
#     return x, y, w, h

# def check_ocr_box(image_source: Union[str, Image.Image], display_img=True, output_bb_format='xywh',
#                  goal_filtering=None, easyocr_args=None, use_paddleocr=False):
#     logger.info(f"Checking OCR boxes, use_paddleocr={use_paddleocr}")
#     try:
#         if isinstance(image_source, str):
#             logger.debug(f"The input file is going to be loaded as PIL image")
#             image_source = Image.open(image_source)
#         if image_source.mode == 'RGBA':
#             logger.debug(f"RGBA conversion is on")
#             image_source = image_source.convert('RGB')
#         image_np = np.array(image_source)
#         w, h = image_source.size
#         logger.debug(f"Image loaded and converted into array")
#         if use_paddleocr:
#             if easyocr_args is None:
#                 text_threshold = 0.5
#             else:
#                 text_threshold = easyocr_args['text_threshold']
#             logger.debug(f"About to fire Padde OCR")
#             result = paddle_ocr.ocr(image_np, cls=False)[0]
#             coord = [item[0] for item in result if item[1][1] > text_threshold]
#             text = [item[1][0] for item in result if item[1][1] > text_threshold]
#         else:
#             if easyocr_args is None:
#                 easyocr_args = {}
#             result = reader.readtext(image_np, **easyocr_args)
#             coord = [item[0] for item in result]
#             text = [item[1] for item in result]
#         logger.debug(f"Reached Display Image")
#         if display_img:
#             opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#             bb = []
#             for item in coord:
#                 x, y, a, b = get_xywh(item)
#                 bb.append((x, y, a, b))
#                 cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
#             plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
#         else:
#             if output_bb_format == 'xywh':
#                 bb = [get_xywh(item) for item in coord]
#             elif output_bb_format == 'xyxy':
#                 bb = [get_xyxy(item) for item in coord]
       
#         logger.success(f"OCR processing completed, found {len(text)} text boxes")
#         return (text, bb), goal_filtering
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         logger.error(f"Error in OCR processing: {str(e)}")
#         raise

from typing import Union, Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- geometry helpers --------------------------------------------------------
def get_xywh(box: np.ndarray | List[List[int]]) -> Tuple[int, int, int, int]:
    x, y, w, h = box[0][0], box[0][1], box[2][0] - box[0][0], box[2][1] - box[0][1]
    return int(x), int(y), int(w), int(h)

def get_xyxy(box: np.ndarray | List[List[int]]) -> Tuple[int, int, int, int]:
    x, y, xp, yp = box[0][0], box[0][1], box[2][0], box[2][1]
    return int(x), int(y), int(xp), int(yp)

def get_xywh_yolo(coords: List[int] | np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]
    return int(x), int(y), int(w), int(h)

# --- main OCR routine --------------------------------------------------------
def check_ocr_box(
    image_source: Union[str, np.ndarray],
    display_img: bool = True,
    output_bb_format: str = "xywh",
    goal_filtering=None,
    easyocr_args: dict | None = None,
    use_paddleocr: bool = False,
):
    """
    Detects text boxes and (optionally) draws them.

    Parameters
    ----------
    image_source : str | np.ndarray
        Path to an image file or an already-loaded BGR/RGB NumPy array.
    display_img : bool, default True
        If True, show the image with bounding boxes using matplotlib.
    output_bb_format : {'xywh', 'xyxy'}, default 'xywh'
        Format of the boxes in the returned tuple when display_img=False.
    goal_filtering, easyocr_args, use_paddleocr
        Passed through exactly like the original implementation.

    Returns
    -------
    (texts, boxes), goal_filtering
    """
    # --- load image with OpenCV ---------------------------------------------
    if isinstance(image_source, str):
        # cv2.imread loads as BGR
        image_np = cv2.imread(image_source, cv2.IMREAD_COLOR)
        if image_np is None:
            raise FileNotFoundError(f"Could not read image: {image_source}")
    else:
        image_np = image_source.copy()

    # OCR libraries expect RGB
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # --- run OCR -------------------------------------------------------------
    if use_paddleocr:
        text_threshold = easyocr_args.get("text_threshold", 0.5) if easyocr_args else 0.5
        result = paddle_ocr.ocr(img_rgb, cls=False)[0]
        boxes = [item[0] for item in result if item[1][1] > text_threshold]
        texts = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(img_rgb, **easyocr_args)
        boxes  = [item[0] for item in result]
        texts  = [item[1] for item in result]

    # --- draw &/or return ----------------------------------------------------
    if display_img:
        drawn = image_np.copy()  # keep BGR for OpenCV drawing
        for box in boxes:
            x, y, w, h = get_xywh(box)
            cv2.rectangle(drawn, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        # caller doesn’t need boxes when we already visualised them
        formatted_boxes = None
    else:
        if output_bb_format == "xywh":
            formatted_boxes = [get_xywh(b) for b in boxes]
        elif output_bb_format == "xyxy":
            formatted_boxes = [get_xyxy(b) for b in boxes]
        else:
            raise ValueError("output_bb_format must be 'xywh' or 'xyxy'")

    return (texts, formatted_boxes), goal_filtering
