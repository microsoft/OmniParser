
def get_caption_model(
    model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None
):
    import torch
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == "cpu":
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float32
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float16
            ).to(device)
    elif model_name == "florence2":
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        if device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
            ).to(device)
    else:
        raise ValueError(f"Unsupported caption model: {model_name}")
    return {"model": model.to(device), "processor": processor}


def get_yolo_model(model_path):
    from ultralytics import YOLO

    return YOLO(model_path)
