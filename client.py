"""
This module provides a command-line interface to interact with the OmniParser Gradio server.

Usage:
    python client.py "http://<server_ip>:7861" "path/to/image.jpg"
"""

import fire
from gradio_client import Client
from loguru import logger
from PIL import Image
import base64
from io import BytesIO
import os
import shutil
import json
from datetime import datetime

def predict(server_url: str, image_path: str, box_threshold: float = 0.05, iou_threshold: float = 0.1):
    """
    Makes a prediction using the OmniParser Gradio client with the provided server URL and image.

    Args:
        server_url (str): The URL of the OmniParser Gradio server.
        image_path (str): Path to the image file to be processed.
        box_threshold (float): Box threshold value (default: 0.05).
        iou_threshold (float): IOU threshold value (default: 0.1).
    """
    client = Client(server_url)
    
    # Generate a timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Prepare the image input in the format expected by the server
    image_input = {
        "path": None,
        "url": f"data:image/png;base64,{encoded_image}",
        "size": None,
        "orig_name": image_path,
        "mime_type": "image/png",
        "is_stream": False,
        "meta": {}
    }

    # Make the prediction
    try:
        result = client.predict(
            image_input,    # image input as dictionary
            box_threshold,  # box_threshold
            iou_threshold,  # iou_threshold
            api_name="/process"
        )

        # Process and log the results
        output_image, result_json = result
        
        logger.info("Prediction completed successfully")

        # Parse the JSON string into a Python object
        result_data = json.loads(result_json)

        # Extract label_coordinates and parsed_content_list
        label_coordinates = result_data['label_coordinates']
        parsed_content_list = result_data['parsed_content_list']

        logger.info(f"{label_coordinates=}")
        logger.info(f"{parsed_content_list=}")

        # Save result data to JSON file
        result_data_path = f"result_data_{timestamp}.json"
        with open(result_data_path, "w") as json_file:
            json.dump(result_data, json_file, indent=4)
        logger.info(f"Parsed content saved to: {result_data_path}")
        
        # Save the output image
        output_image_path = f"output_image_{timestamp}.png"
        if isinstance(output_image, str) and os.path.exists(output_image):
            shutil.copy(output_image, output_image_path)
            logger.info(f"Output image saved to: {output_image_path}")
        else:
            logger.warning(f"Unexpected output_image format or file not found: {output_image}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Traceback:")

if __name__ == "__main__":
    fire.Fire(predict)

