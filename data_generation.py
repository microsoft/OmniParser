import os
import json
from pathlib import Path
import cv2
import numpy as np
import base64
import time
import torch
from ultralytics import YOLO
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from loguru import logger

def get_unique_filename(target_path, filename):
    """Generate a unique filename by appending a counter if the file exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(target_path, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

def initialize_models():
    """Initialize YOLO and Florence2 models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'weights/icon_detect/model.pt'
    caption_model_path = 'weights/icon_caption_florence'
    
    try:
        # Initialize YOLO model
        som_model = get_yolo_model(model_path)
        som_model.to(device)
        logger.info(f'YOLO model loaded to {device}')
        
        # Initialize caption model (Florence2)
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path=caption_model_path, 
            device=device
        )
        logger.info(f'Florence2 caption model loaded to {device}')
        
        return som_model, caption_model_processor
    except Exception as e:
        logger.info(f'Error initializing models: {str(e)}')
        return None, None

def cv2_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def process_image(image_path, step_dir, som_model, caption_model_processor):
    """Process a single image and save annotated image and parsed content."""
    try:
        start = time.time()
        # Load image using OpenCV
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.info(f'Error: Could not load image {image_path}')
            return
            
        # Convert BGR to RGB for processing (models expect RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        logger.info(f'Processing image: {image_path}, size: {width}x{height}')

        # Configure bounding box drawing
        box_overlay_ratio = max(width, height) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        BOX_THRESHOLD = 0.05

        # Perform OCR
        start = time.time()
        ocr_bbox_rslt, _ = check_ocr_box(
            image_path, display_img=False, output_bb_format='xyxy',
            goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        # ocr_bbox_rslt, _ = check_ocr_box(
        #     image_rgb, display_img=False, output_bb_format='xyxy',
        #     goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        #     use_paddleocr=True
        # )
        text, ocr_bbox = ocr_bbox_rslt

        # Process image with SOM model
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path, som_model, BOX_TRESHOLD=BOX_THRESHOLD, output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor, ocr_text=text,
            use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128
        )
        # dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        #     image_rgb, som_model, BOX_TRESHOLD=BOX_THRESHOLD, output_coord_in_ratio=True,
        #     ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config,
        #     caption_model_processor=caption_model_processor, ocr_text=text,
        #     use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128
        # )

        # Delete OCR results to free memory
        del ocr_bbox_rslt, text, ocr_bbox
        # Delete image arrays as they're no longer needed
        del image_bgr, image_rgb

        # Save annotated image using OpenCV
        image_name = os.path.basename(image_path)
        annotated_filename = f"annotated_{image_name}"
        annotated_path = os.path.join(step_dir, get_unique_filename(step_dir, annotated_filename))
        
        # Convert base64 back to OpenCV image and save
        annotated_image = base64_to_cv2(dino_labled_img)
        cv2.imwrite(annotated_path, annotated_image)
        logger.info(f'Saved annotated image: {annotated_path}')

        # Delete annotation data to free memory
        del dino_labled_img, annotated_image
        # Delete label_coordinates if not needed
        del label_coordinates

        # Save parsed content as JSON
        json_filename = f"json_{image_name[:-4]}.json"
        json_path = os.path.join(step_dir, get_unique_filename(step_dir, json_filename))
        with open(json_path, 'w') as f:
            json.dump(parsed_content_list, f, indent=2)
        logger.info(f'Saved JSON: {json_path}')

        # Delete parsed_content_list to free memory
        del parsed_content_list
        logger.info(f"TOOK: {round(time.time()-start)} Secs to process DIRECTORY: {step_dir}")
    except Exception as e:
        logger.info(f'Error processing {image_path}: {str(e)}')
        
def process_case_directory(case_dir, som_model, caption_model_processor):
    """Process .jpg files in step_* directories within a case directory."""
    # Get all step_* directories
    step_dirs = [d for d in os.listdir(case_dir) if d.startswith('step_') and os.path.isdir(os.path.join(case_dir, d))]
    
    # Sort by step number (extract numeric part after 'step_')
    def get_step_number(step_dir):
        try:
            return int(step_dir.split('_')[1])
        except (ValueError, IndexError):
            return float('inf')  # Put invalid names at the end
    
    step_dirs.sort(key=get_step_number)  # Sort by step number

    for step_dir in step_dirs:
        skip_flag = False
        step_path = os.path.join(case_dir, step_dir)
        # Process .jpg files in the step directory (excluding annotated_*.jpg)
        for file in os.listdir(os.path.join(case_dir, step_dir)):
            if "annotated_" in file:
                skip_flag = True
        if skip_flag:
            logger.warning(f"DIRECTORY: {step_dir} ALREADY PROCESSED AND SKIPPING")
            skip_flag = False 
            continue
        for file in os.listdir(step_path):
            if file.endswith('.jpg') and not file.startswith('annotated_'):
                image_path = os.path.join(step_path, file)
                process_image(image_path, step_path, som_model, caption_model_processor)

def main():
    """Process case_* directories from case_a to case_b within the root directory."""
    root_dir = 'data/UserEvents'  # Hardcoded root directory
    
    # Hardcoded variables
    a = 1
    b = 100
    
    if not os.path.isdir(root_dir):
        logger.info(f"Error: The root directory '{root_dir}' does not exist.")
        return

    logger.info(f"Processing case directories from case_{a} to case_{b}")

    # Initialize models
    som_model, caption_model_processor = initialize_models()
    if som_model is None or caption_model_processor is None:
        logger.info("Failed to initialize models. Exiting.")
        return

    # Get all case_* directories and filter by range a to b
    case_dirs = [d for d in os.listdir(root_dir) if d.startswith('case_') and os.path.isdir(os.path.join(root_dir, d))]
    case_dirs.sort()  # Sort for consistent processing
    
    # Filter case directories to process only from case_a to case_b
    filtered_case_dirs = []
    for case_dir in case_dirs:
        try:
            # Extract the case number from directory name (e.g., case_4_e... -> 4)
            case_number = int(case_dir.split('_')[1])
            if a <= case_number <= b:
                filtered_case_dirs.append(case_dir)
        except (ValueError, IndexError):
            # Skip directories that don't follow the expected naming pattern
            continue

    if not filtered_case_dirs:
        logger.info(f"No case directories found in range case_{a} to case_{b}")
        return

    logger.info(f"Found {len(filtered_case_dirs)} case directories to process:")
    for case_dir in filtered_case_dirs:
        logger.info(f"  - {case_dir}")

    for case_dir in filtered_case_dirs:
        logger.info(f"Processing directory: {case_dir}")
        process_case_directory(os.path.join(root_dir, case_dir), som_model, caption_model_processor)

    logger.info("Processing complete.")

if __name__ == "__main__":
    main()