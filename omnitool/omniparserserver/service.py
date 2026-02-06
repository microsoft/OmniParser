import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from omniparser_core.omniparser import Omniparser

logger = logging.getLogger(__name__)


class OmniParserService:
    def __init__(self, model_config: dict[str, Any]) -> None:
        self._parser = Omniparser(model_config)

    def parse_image(self, base64_image: str) -> tuple[str, list[dict[str, Any]], float]:
        logger.info("Start parsing request")
        start = time.time()
        som_image_base64, parsed_content_list = self._parser.parse(base64_image)
        latency = time.time() - start
        logger.info("Completed parsing request in %.3fs", latency)
        return som_image_base64, parsed_content_list, latency
