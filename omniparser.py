from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from PIL import Image
from typing import Dict, Tuple, List, Any
import io
import base64
import json
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

config = {
    'som_model_path': 'finetuned_icon_detect.pt',
    'device': 'cpu',
    'caption_model_path': 'Salesforce/blip2-opt-2.7b',
    'draw_bbox_config': {
        'text_scale': 0.8,
        'text_thickness': 2,
        'text_padding': 3,
        'thickness': 3,
    },
    'BOX_THRESHOLD': 0.05  # Fixed spelling
}

class Omniparser:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.som_model = get_yolo_model(model_path=config['som_model_path'])

    def parse(self, image_path: str) -> Tuple[Image.Image, List[Dict]]:
        logging.info(f'Parsing image: {image_path}')
        try:
            ocr_bbox_result, is_goal_filtered = check_ocr_box(
                image_path, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.9}
            )
            text, ocr_bbox = ocr_bbox_result

            draw_bbox_config = self.config['draw_bbox_config']
            BOX_THRESHOLD = self.config['BOX_THRESHOLD']
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                image_path, 
                self.som_model, 
                BOX_THRESHOLD=BOX_THRESHOLD, 
                output_coord_in_ratio=False, 
                ocr_bbox=ocr_bbox, 
                draw_bbox_config=draw_bbox_config, 
                caption_model_processor=None, 
                ocr_text=text, 
                use_local_semantics=False
            )

            image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))

            # Formatting output
            return_list = [
                {'from': 'omniparser', 
                 'shape': {'x': coord[0], 'y': coord[1], 'width': coord[2], 'height': coord[3]},
                 'text': parsed_content_list[i].split(': ')[1], 
                 'type': 'text'}
                for i, (k, coord) in enumerate(label_coordinates.items()) if i < len(parsed_content_list)
            ]
            return_list.extend(
                [{'from': 'omniparser', 
                  'shape': {'x': coord[0], 'y': coord[1], 'width': coord[2], 'height': coord[3]},
                  'text': 'None', 'type': 'icon'}
                 for i, (k, coord) in enumerate(label_coordinates.items()) if i >= len(parsed_content_list)]
            )

            return [image, return_list]
        except Exception as e:
            logging.error(f"Error during parsing: {e}")
            raise

    def parse_batch(self, image_paths: List[str]) -> List[Tuple[Image.Image, List[Dict]]]:
        results = []
        for image_path in image_paths:
            result = self.parse(image_path)
            results.append(result)
        return results

    def export_results(self, results: List[Tuple[Image.Image, List[Dict]]], output_file: str, format: str = 'json'):
        try:
            if format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(results, f, default=str)  # Ensure proper JSON serialization
                logging.info(f"Results exported to {output_file} in JSON format.")
            elif format == 'csv':
                # Implement CSV export logic if needed
                logging.warning("CSV export not implemented.")
            else:
                logging.error("Unsupported format. Use 'json' or 'csv'.")
        except Exception as e:
            logging.error(f"Error during export: {e}")
            raise

# Example usage
if __name__ == "__main__":
    parser = Omniparser(config)
    image_path = 'examples/pc_1.png'

    # Time the parser
    start_time = time.time()
    image, parsed_content_list = parser.parse(image_path)
    device = config['device']
    logging.info(f'Time taken for Omniparser on {device}: {time.time() - start_time:.2f} seconds')

    # Optionally, you can export the results
    results = [(image, parsed_content_list)]
    parser.export_results(results, 'output_results.json')
