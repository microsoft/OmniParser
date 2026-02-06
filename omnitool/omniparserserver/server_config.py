import argparse
from dataclasses import dataclass


@dataclass
class ServerSettings:
    som_model_path: str
    caption_model_name: str
    caption_model_path: str
    device: str
    box_threshold: float
    host: str
    port: int
    reload: bool

    def to_model_config(self) -> dict:
        return {
            "som_model_path": self.som_model_path,
            "caption_model_name": self.caption_model_name,
            "caption_model_path": self.caption_model_path,
            "device": self.device,
            "BOX_TRESHOLD": self.box_threshold,
        }


def parse_cli_args() -> ServerSettings:
    parser = argparse.ArgumentParser(description="OmniParser API")
    parser.add_argument(
        "--som_model_path",
        type=str,
        default="../../weights/icon_detect/model.pt",
        help="Path to the icon detection model",
    )
    parser.add_argument(
        "--caption_model_name",
        type=str,
        default="florence2",
        help="Caption model name",
    )
    parser.add_argument(
        "--caption_model_path",
        type=str,
        default="../../weights/icon_caption_florence",
        help="Path to the caption model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--BOX_TRESHOLD",
        type=float,
        default=0.05,
        help="Detection threshold for icon boxes",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload mode",
    )
    args = parser.parse_args()
    return ServerSettings(
        som_model_path=args.som_model_path,
        caption_model_name=args.caption_model_name,
        caption_model_path=args.caption_model_path,
        device=args.device,
        box_threshold=args.BOX_TRESHOLD,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
