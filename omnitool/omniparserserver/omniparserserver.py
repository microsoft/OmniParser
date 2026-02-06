"""
Run:
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
"""

import logging

import uvicorn

from app import create_app
from server_config import parse_cli_args
from service import OmniParserService

logging.basicConfig(level=logging.INFO)

settings = parse_cli_args()
service = OmniParserService(settings.to_model_config())
app = create_app(service)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.reload)
