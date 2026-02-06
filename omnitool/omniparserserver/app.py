from fastapi import FastAPI

from .schemas import ParseRequest, ParseResponse, ProbeResponse
from .service import OmniParserService


def create_app(parser_service: OmniParserService) -> FastAPI:
    app = FastAPI(title="OmniParser API", version="2.0")

    @app.post("/parse/", response_model=ParseResponse)
    async def parse_endpoint(parse_request: ParseRequest) -> ParseResponse:
        som_image_base64, parsed_content_list, latency = parser_service.parse_image(
            parse_request.base64_image
        )
        return ParseResponse(
            som_image_base64=som_image_base64,
            parsed_content_list=parsed_content_list,
            latency=latency,
        )

    @app.get("/probe/", response_model=ProbeResponse)
    async def probe_endpoint() -> ProbeResponse:
        return ProbeResponse(message="Omniparser API ready")

    return app
