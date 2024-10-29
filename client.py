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
        output_image, parsed_content = result
        
        logger.info("Prediction completed successfully")
        logger.info(f"Parsed content:\n{parsed_content}")
        
        # Save the output image
        output_image_path = "output_image.png"
        if isinstance(output_image, dict) and 'url' in output_image:
            # Handle base64 encoded image
            img_data = base64.b64decode(output_image['url'].split(',')[1])
            with open(output_image_path, 'wb') as f:
                f.write(img_data)
        elif isinstance(output_image, str):
            if output_image.startswith('data:image'):
                # Handle base64 encoded image string
                img_data = base64.b64decode(output_image.split(',')[1])
                with open(output_image_path, 'wb') as f:
                    f.write(img_data)
            elif os.path.exists(output_image):
                # Handle file path
                shutil.copy(output_image, output_image_path)
            else:
                logger.warning(f"Unexpected output_image format: {output_image}")
        elif isinstance(output_image, Image.Image):
            output_image.save(output_image_path)
        else:
            logger.warning(f"Unexpected output_image format: {type(output_image)}")
            logger.warning(f"Output image content: {output_image[:100]}...")  # Log the first 100 characters
        
        if os.path.exists(output_image_path):
            logger.info(f"Output image saved to: {output_image_path}")
        else:
            logger.warning(f"Failed to save output image to: {output_image_path}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Traceback:")

if __name__ == "__main__":
    fire.Fire(predict)

