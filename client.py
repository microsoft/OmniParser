"""
This module provides a command-line interface and programmatic API to interact with the OmniParser Gradio server.

Command-line usage:
    python client.py "http://<server_ip>:7861" "path/to/image.jpg"

View results:
    JSON: cat result_data_<timestamp>.json
    Image:
        macOS:   open output_image_<timestamp>.png
        Windows: start output_image_<timestamp>.png
        Linux:   xdg-open output_image_<timestamp>.png

Programmatic usage:
    from omniparse.client import predict
    result = predict("http://<server_ip>:7861", "path/to/image.jpg")

Result data format:
    {
        "label_coordinates": {
            "0": [x1, y1, width, height],  // Normalized coordinates for each bounding box
            "1": [x1, y1, width, height],
            ...
        },
        "parsed_content_list": [
            "Text Box ID 0: [content]",
            "Text Box ID 1: [content]",
            ...,
            "Icon Box ID X: [description]",
            ...
        ]
    }

Note: The parsed_content_list includes both text box contents and icon descriptions.
"""

import fire
from gradio_client import Client
from loguru import logger
import base64
import os
import shutil
import json
from datetime import datetime

# Define constants for default thresholds
DEFAULT_BOX_THRESHOLD = 0.05
DEFAULT_IOU_THRESHOLD = 0.1

def predict(server_url: str, image_path: str, box_threshold: float = DEFAULT_BOX_THRESHOLD, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
    """
    Makes a prediction using the OmniParser Gradio client with the provided server URL and image.
    Args:
        server_url (str): The URL of the OmniParser Gradio server.
        image_path (str): Path to the image file to be processed.
        box_threshold (float): Box threshold value (default: 0.05).
        iou_threshold (float): IOU threshold value (default: 0.1).
    Returns:
        dict: Parsed result data containing label coordinates and parsed content list.
    """
    client = Client(server_url)

    # Load and encode the image
    image_path = os.path.expanduser(image_path)
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
    result = client.predict(
        image_input,
        box_threshold,
        iou_threshold,
        api_name="/process"
    )

    # Process and return the result
    output_image, result_json = result
    result_data = json.loads(result_json)

    return {"output_image": output_image, "result_data": result_data}


def predict_and_save(server_url: str, image_path: str, box_threshold: float = DEFAULT_BOX_THRESHOLD, iou_threshold: float = DEFAULT_IOU_THRESHOLD):
    """
    Makes a prediction and saves the results to files, including logs and image outputs.
    Args:
        server_url (str): The URL of the OmniParser Gradio server.
        image_path (str): Path to the image file to be processed.
        box_threshold (float): Box threshold value (default: 0.05).
        iou_threshold (float): IOU threshold value (default: 0.1).
    """
    # Generate a timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Call the predict function to get prediction data
    try:
        result = predict(server_url, image_path, box_threshold, iou_threshold)
        output_image = result["output_image"]
        result_data = result["result_data"]

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
    fire.Fire(predict_and_save)
