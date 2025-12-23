import numpy as np
import cv2

from .models import COCO, SegmentedImage


def segment_image_open_cv(image: np.ndarray) -> SegmentedImage:
    height, width = image.shape[:2]
    
    coco = COCO()
    coco.images.append({
        "id": 1,
        "width": width,
        "height": height
    })
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = (width * height) * 0.005
    max_area = (width * height) * 0.95
    
    annotation_id = 1
    category_id = 1
    
    coco.categories.append({
        "id": category_id,
        "name": "image"
    })
    
    detected_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            continue
        
        extent = area / (w * h) if (w * h) > 0 else 0
        if extent < 0.4:
            continue
        
        roi = gray[y:y+h, x:x+w] if len(gray.shape) == 2 else image[y:y+h, x:x+w]
        
        if roi.size == 0:
            continue
        
        std_dev = np.std(roi)
        if std_dev < 10:
            continue
        
        detected_regions.append((x, y, w, h, area))
    
    detected_regions.sort(key=lambda r: r[4], reverse=True)
    
    for x, y, w, h, area in detected_regions:
        coco.annotations.append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0
        })
        annotation_id += 1
    
    return SegmentedImage(image=image, segmentation=coco)

