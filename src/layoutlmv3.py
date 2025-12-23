import numpy as np
import torch
from PIL import Image
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from .models import COCO, SegmentedImage


def segment_image_layoutlmv3(image: np.ndarray) -> SegmentedImage:
    pil_image = Image.fromarray(image.astype(np.uint8))
    
    ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    
    img_width, img_height = pil_image.size
    
    words: list[str] = []
    boxes: list[list[int]] = []
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            words.append(text)
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            x_norm = int((x / img_width) * 1000)
            y_norm = int((y / img_height) * 1000)
            x2_norm = int(((x + w) / img_width) * 1000)
            y2_norm = int(((y + h) / img_height) * 1000)
            
            boxes.append([x_norm, y_norm, x2_norm, y2_norm])
    
    if not words:
        coco = COCO()
        coco.images.append({
            "id": 1,
            "width": image.shape[1],
            "height": image.shape[0]
        })
        return SegmentedImage(image=image, segmentation=coco)
    
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    
    encoding = processor(
        pil_image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    
    id2label = model.config.id2label
    predicted_labels = [id2label[p.item()] for p in predictions[0]]
    
    height, width = image.shape[:2]
    
    coco = COCO()
    coco.images.append({
        "id": 1,
        "width": width,
        "height": height
    })
    
    label_to_category = {}
    annotation_id = 1
    
    def add_annotation(label: str, indices: list[int]) -> None:
        nonlocal annotation_id
        category_name = label.replace('B-', '').replace('I-', '')
        if category_name not in label_to_category:
            category_id = len(coco.categories) + 1
            label_to_category[category_name] = category_id
            coco.categories.append({
                "id": category_id,
                "name": category_name
            })
        else:
            category_id = label_to_category[category_name]
        
        all_boxes = [boxes[idx] for idx in indices]
        x_min = min(box[0] for box in all_boxes)
        y_min = min(box[1] for box in all_boxes)
        x_max = max(box[2] for box in all_boxes)
        y_max = max(box[3] for box in all_boxes)
        
        coco.annotations.append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": category_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        })
        annotation_id += 1
    
    current_label = None
    current_indices = []
    
    for i, (word, box, label) in enumerate(zip(words, boxes, predicted_labels[:len(words)])):
        if label.startswith('B-') or (current_label is None and label != 'O'):
            if current_label is not None and current_indices:
                add_annotation(current_label, current_indices)
            
            if label != 'O':
                current_label = label
                current_indices = [i]
            else:
                current_label = None
                current_indices = []
        elif label.startswith('I-') and current_label and label.replace('I-', '') == current_label.replace('B-', '').replace('I-', ''):
            current_indices.append(i)
        else:
            if current_label is not None and current_indices:
                add_annotation(current_label, current_indices)
            
            current_label = None
            current_indices = []
    
    if current_label is not None and current_indices:
        add_annotation(current_label, current_indices)
    
    return SegmentedImage(image=image, segmentation=coco)

