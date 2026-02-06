import torch


def predict(model, image, caption, box_threshold, text_threshold):
    model, processor = model["model"], model["processor"]
    device = model.device
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    if scale_img:
        result = model.predict(source=image, conf=box_threshold, imgsz=imgsz, iou=iou_threshold)
    else:
        result = model.predict(source=image, conf=box_threshold, iou=iou_threshold)
    boxes = result[0].boxes.xyxy
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]
    return boxes, conf, phrases
