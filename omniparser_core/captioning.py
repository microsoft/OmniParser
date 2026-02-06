import time

import cv2
import torch
from torchvision.transforms import ToPILImage


@torch.inference_mode()
def get_parsed_content_icon(
    filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128
):
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    cropped_pil_images = []
    for coord in non_ocr_boxes:
        try:
            xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
            ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            cropped_pil_images.append(to_pil(cropped_image))
        except Exception:
            continue

    model, processor = caption_model_processor["model"], caption_model_processor["processor"]
    if not prompt:
        if "florence" in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    generated_texts = []
    device = model.device
    for i in range(0, len(cropped_pil_images), batch_size):
        batch = cropped_pil_images[i : i + batch_size]
        if model.device.type == "cuda":
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt", do_resize=False
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(
                device=device
            )
        if "florence" in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts.extend([gen.strip() for gen in generated_text])

    return generated_texts


def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox) :]
    else:
        non_ocr_boxes = filtered_boxes

    cropped_pil_images = []
    for coord in non_ocr_boxes:
        xmin, xmax = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
        ymin, ymax = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        cropped_pil_images.append(to_pil(cropped_image))

    model, processor = caption_model_processor["model"], caption_model_processor["processor"]
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5
    generated_texts = []
    for i in range(0, len(cropped_pil_images), batch_size):
        images = cropped_pil_images[i : i + batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs = {"input_ids": [], "attention_mask": [], "pixel_values": [], "image_sizes": []}
        texts = [prompt] * len(images)
        for idx, txt in enumerate(texts):
            inp = processor._convert_images_texts_to_inputs(image_inputs[idx], txt, return_tensors="pt")
            inputs["input_ids"].append(inp["input_ids"])
            inputs["attention_mask"].append(inp["attention_mask"])
            inputs["pixel_values"].append(inp["pixel_values"])
            inputs["image_sizes"].append(inp["image_sizes"])
        max_len = max([x.shape[1] for x in inputs["input_ids"]])
        for idx, value in enumerate(inputs["input_ids"]):
            inputs["input_ids"][idx] = torch.cat(
                [processor.tokenizer.pad_token_id * torch.ones(1, max_len - value.shape[1], dtype=torch.long), value],
                dim=1,
            )
            inputs["attention_mask"][idx] = torch.cat(
                [torch.zeros(1, max_len - value.shape[1], dtype=torch.long), inputs["attention_mask"][idx]],
                dim=1,
            )
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = {"max_new_tokens": 25, "temperature": 0.01, "do_sample": False}
        generate_ids = model.generate(
            **inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
        )
        generate_ids = generate_ids[:, inputs_cat["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generated_texts.extend([res.strip("\n").strip() for res in response])

    return generated_texts
