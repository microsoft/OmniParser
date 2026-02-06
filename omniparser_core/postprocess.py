from typing import List


def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
    return (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)

    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem["bbox"]
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            box2 = box2_elem["bbox"]
            if i != j and iou(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                box_added = False
                ocr_labels = ""
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem["bbox"]
                        if is_inside(box3, box1):
                            try:
                                ocr_labels += box3_elem["content"] + " "
                                filtered_boxes.remove(box3_elem)
                            except Exception:
                                continue
                        elif is_inside(box1, box3):
                            box_added = True
                            break
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append(
                            {
                                "type": "icon",
                                "bbox": box1_elem["bbox"],
                                "interactivity": True,
                                "content": ocr_labels,
                                "source": "box_yolo_content_ocr",
                            }
                        )
                    else:
                        filtered_boxes.append(
                            {
                                "type": "icon",
                                "bbox": box1_elem["bbox"],
                                "interactivity": True,
                                "content": None,
                                "source": "box_yolo_content_yolo",
                            }
                        )
            else:
                filtered_boxes.append(box1)
    return filtered_boxes
