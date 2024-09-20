# from ultralytics import YOLO
import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory # 'ch_sim',
import time
import base64

import os
import ast
import torch
from typing import Tuple, List
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T


def get_caption_model_processor(model_name="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "Salesforce/blip2-opt-2.7b":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", device_map=None, torch_dtype=torch.float16
            # '/home/yadonglu/sandbox/data/orca/blipv2_ui_merge', device_map=None, torch_dtype=torch.float16
        )  
    elif model_name == "blip2-opt-2.7b-ui":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
            '/home/yadonglu/sandbox/data/orca/blipv2_ui_merge', device_map=None, torch_dtype=torch.float32
        ) 
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
            '/home/yadonglu/sandbox/data/orca/blipv2_ui_merge', device_map=None, torch_dtype=torch.float16
        )
    elif model_name == "florence":
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained("/home/yadonglu/sandbox/data/orca/florence-2-base-ft-fft_ep1_rai", torch_dtype=torch.float32, trust_remote_code=True)#.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained("/home/yadonglu/sandbox/data/orca/florence-2-base-ft-fft_ep1_rai_win_ep5_fixed", torch_dtype=torch.float16, trust_remote_code=True).to(device)
    elif model_name == 'phi3v_ui':
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        model = AutoModelForCausalLM.from_pretrained('/home/yadonglu/sandbox/data/orca/phi3v_ui', device_map=device, trust_remote_code=True, torch_dtype="auto")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
    elif model_name == 'phi3v':
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, trust_remote_code=True, torch_dtype="auto")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return {'model': model.to(device), 'processor': processor}


def get_yolo_model():
    from ultralytics import YOLO
    # Load the model.
    # model = YOLO('/home/yadonglu/sandbox/data/yolo/runs/detect/yolov8n_v8_xcyc/weights/best.pt')
    model = YOLO('/home/yadonglu/sandbox/data/yolo/runs/detect/yolov8n_v8_seq_xcyc_b32_n4_office_ep20/weights/best.pt')
    return model


def get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=None):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    # import pdb; pdb.set_trace()
    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
            # prompt = "NO gender!NO gender!NO gender! The image shows a icon:"

    batch_size = 10  # Number of samples per batch
    generated_texts = []
    device = model.device

    for i in range(0, len(croped_pil_image), batch_size):
        batch = croped_pil_image[i:i+batch_size]
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=1024,num_beams=3, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
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

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                if not any(IoU(box1, box3) > iou_threshold for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    from util.box_annotator import BoxAnnotator 
    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image_path, box_threshold):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    
    result = model.predict(
    source=image_path,
    conf=box_threshold,
    # iou=0.5, # default 0.7
    )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def get_som_labeled_img(img_path, model=None, BOX_TRESHOLD = 0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None):
    """ ocr_bbox: list of xyxy format bbox
    """
    TEXT_PROMPT = "clickable buttons on the screen"
    # BOX_TRESHOLD = 0.02 # 0.05/0.02 for web and 0.1 for mobile
    TEXT_TRESHOLD = 0.01 # 0.9 # 0.01
    image_source = Image.open(img_path).convert("RGB")
    w, h = image_source.size
    # import pdb; pdb.set_trace()
    if False: # TODO
        xyxy, logits, phrases = predict(model=model, image=image_source, caption=TEXT_PROMPT, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD)
    else:
        xyxy, logits, phrases = predict_yolo(model=model, image_path=img_path, box_threshold=BOX_TRESHOLD)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    h, w, _ = image_source.shape
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None
    filtered_boxes = remove_overlap(boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox)
    
    # get parsed icon local semantics
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, ocr_bbox, image_source, caption_model_processor, prompt=prompt)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]
    
    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        # h, w, _ = image_source.shape
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, parsed_content_merged


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h
    

def run_api(body, max_tokens=1024):
    '''
    API call, check https://platform.openai.com/docs/guides/vision for the latest api usage. 
    '''
    max_num_trial = 3
    num_trial = 0
    while num_trial < max_num_trial:
        try:
            response = client.chat.completions.create(
                                    model=deployment,
                                    messages=body,
                                    temperature=0.01,
                                    max_tokens=max_tokens,
                                )
            return response.choices[0].message.content
        except:
            print('retry call gptv', num_trial)
            num_trial += 1
            time.sleep(10)
    return ''

def call_gpt4v_new(message_text, image_path=None, max_tokens=2048):
    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('ascii')
        except: 
            encoded_image = image_path
    
    if image_path:
        content = [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}, {"type": "text","text": message_text},]
    else:
        content = [{"type": "text","text": message_text},]

    max_num_trial = 3
    num_trial = 0
    call_api_success = True

    while num_trial < max_num_trial:
        try:
            response = client.chat.completions.create(
                            model=deployment,
                            messages=[
                                        {
                                        "role": "system",
                                        "content": [
                                            {
                                            "type": "text",
                                            "text": "You are an AI assistant that is good at making plans and analyzing screens, and helping people find information."
                                            },
                                        ]
                                        },
                                        {
                                        "role": "user",
                                        "content": content
                                        }
                                    ],
                            temperature=0.01,
                            max_tokens=max_tokens,
                        )
            ans_1st_pass = response.choices[0].message.content
            break
        except:
            print('retry call gptv', num_trial)
            num_trial += 1
            ans_1st_pass = ''
            time.sleep(10)
    if num_trial == max_num_trial:
        call_api_success = False
    return ans_1st_pass, call_api_success


def check_ocr_box(image_path, display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None):
    if easyocr_args is None:
        easyocr_args = {}
    result = reader.readtext(image_path, **easyocr_args)
    is_goal_filtered = False
    if goal_filtering:
        ocr_filter_fs = "Example 1:\n Based on task and ocr results, ```In summary, the task related bboxes are: [([[3060, 111], [3135, 111], [3135, 141], [3060, 141]], 'Share', 0.949013667261589), ([[3068, 197], [3135, 197], [3135, 227], [3068, 227]], 'Link _', 0.3567054243152049), ([[3006, 321], [3178, 321], [3178, 354], [3006, 354]], 'Manage Access', 0.8800734456437066)] ``` \n Example 2:\n Based on task and ocr results, ```In summary, the task related bboxes are: [([[3060, 111], [3135, 111], [3135, 141], [3060, 141]], 'Search Google or type a URL', 0.949013667261589)] ```"
        # message_text = f"Based on the ocr results which contains text+bounding box in a dictionary, please filter it so that it only contains the task related bboxes. The task is: {goal_filtering}, the ocr results are: {str(result)}. Your final answer should be in the exact same format as the ocr results, please do not include any other redundant information, please do not include any analysis."
        message_text = f"Based on the task and ocr results which contains text+bounding box in a dictionary, please filter it so that it only contains the task related bboxes.  Requirement: 1. first give a brief analysis. 2. provide an answer in the format: ```In summary, the task related bboxes are: ..```, you must put it inside ``` ```.  Do not include any info after ```.\n {ocr_filter_fs}\n The task is: {goal_filtering}, the ocr results are: {str(result)}."

        prompt = [{"role":"system", "content": "You are an AI assistant that helps people find the correct way to operate computer or smartphone."}, {"role":"user","content": message_text},]
        print('[Perform OCR filtering by goal] ongoing ...')
        # pred, _, _ = call_gpt4(prompt)
        pred, _, = call_gpt4v(message_text)
        # import pdb; pdb.set_trace()
        try:
            # match = re.search(r"```(.*?)```", pred, re.DOTALL)
            # result = match.group(1).strip()
            # pred = result.split('In summary, the task related bboxes are:')[-1].strip()
            pred = pred.split('In summary, the task related bboxes are:')[-1].strip().strip('```')
            result = ast.literal_eval(pred)
            print('[Perform OCR filtering by goal] success!!! Filtered buttons: ', pred)
            is_goal_filtered = True
        except:
            print('[Perform OCR filtering by goal] failed or unused!!!')
            pass
            # added_prompt = [{"role":"assistant","content":pred},
            #        {"role":"user","content": "given the previous answers, please provide the final answer in the exact same format as the ocr results, please do not include any other redundant information, please do not include any analysis."}]
            # prompt.extend(added_prompt)
            # pred, _, _ = call_gpt4(prompt)
            # print('goal filtering pred 2nd:', pred)
            # result = ast.literal_eval(pred)
    # print('goal filtering pred:', result[-5:])
    coord = [item[0] for item in result]
    text = [item[1] for item in result]
    # confidence = [item[2] for item in result]
    # if confidence_filtering:
    #     coord = [coord[i] for i in range(len(coord)) if confidence[i] > confidence_filtering]
    #     text = [text[i] for i in range(len(text)) if confidence[i] > confidence_filtering]
    # read the image using cv2
    if display_img:
        opencv_img = cv2.imread(image_path)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            # print(x, y, a, b)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        
        # Display the image
        plt.imshow(opencv_img)
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
        # print('bounding box!!!', bb)
    return (text, bb), is_goal_filtered


def get_pred_gptv(message_text, yolo_labled_img, label_coordinates, summarize_history=True, verbose=True, history=None, id_key='Click ID'):
    """ This func first 
    1. call gptv(yolo_labled_img, text bbox+task) -> ans_1st_cal
    2. call gpt4(ans_1st_cal, label_coordinates) -> final ans
    """

    # Configuration
    encoded_image = yolo_labled_img
    
    # Payload for the request
    if not history:
        messages = [
            {"role": "system", "content": [{"type": "text","text": "You are an AI assistant that is great at interpreting screenshot and predict action."},]},
            {"role": "user","content": [{"type": "text","text": message_text}, {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},]}
            ]
    else:
        messages = [
            {"role": "system", "content": [{"type": "text","text": "You are an AI assistant that is great at interpreting screenshot and predict action."},]},
            history,
            {"role": "user","content": [{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},{"type": "text","text": message_text},]}
            ]

    payload = {
        "messages": messages,
        "temperature": 0.01, # 0.01
        "top_p": 0.95,
        "max_tokens": 800
        }

    max_num_trial = 3
    num_trial = 0
    call_api_success = True
    while num_trial < max_num_trial:
        try:
            # response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
            # response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            # ans_1st_pass = response.json()['choices'][0]['message']['content']
            response = client.chat.completions.create(
                            model=deployment,
                            messages=messages,
                            temperature=0.01,
                            max_tokens=512,
                        )
            ans_1st_pass = response.choices[0].message.content
            break
        except requests.RequestException as e:
            print('retry call gptv', num_trial)
            num_trial += 1
            ans_1st_pass = ''
            time.sleep(30)
            # raise SystemExit(f"Failed to make the request. Error: {e}")
    if num_trial == max_num_trial:
        call_api_success = False
    if verbose:
        print('Answer by GPTV: ', ans_1st_pass)
    # extract by simple parsing
    try: 
        match = re.search(r"```(.*?)```", ans_1st_pass, re.DOTALL)
        if match:
            result = match.group(1).strip()
            pred = result.split('In summary, the next action I will perform is:')[-1].strip().replace('\\', '')
            pred = ast.literal_eval(pred)
        else:
            pred = ans_1st_pass.split('In summary, the next action I will perform is:')[-1].strip().replace('\\', '')
            pred = ast.literal_eval(pred)

        if id_key in pred:
            icon_id = pred[id_key]
            bbox = label_coordinates[str(icon_id)]
            pred['click_point'] = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
    except:
        # import pdb; pdb.set_trace()
        print('gptv action regex extract fail!!!')
        print('ans_1st_pass:', ans_1st_pass)
        pred = {'action_type': 'CLICK', 'click_point': [0, 0], 'value': 'None', 'is_completed': False}

    step_pred_summary = None
    if summarize_history:
        step_pred_summary, _ = call_gpt4v_new('Summarize what action you decide to perform in the current step, in one sentence, and do not include any icon box number: ' + ans_1st_pass, max_tokens=128)
        print('step_pred_summary', step_pred_summary)
    return pred, [call_api_success, ans_1st_pass, None, step_pred_summary]
    # return pred, [call_api_success, message_2nd, completion_2nd.choices[0].message.content, step_pred_summary]


