from typing import Any

from pydantic import BaseModel


class ParseRequest(BaseModel):
    base64_image: str


class ParseResponse(BaseModel):
    som_image_base64: str
    parsed_content_list: list[dict[str, Any]]
    latency: float


class ProbeResponse(BaseModel):
    message: str
