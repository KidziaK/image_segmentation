import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from .models import COCO, SegmentedImage


def segment_image_yolo(image: np.ndarray) -> SegmentedImage:
    h, w = image.shape[:2]
    coco = COCO()
    coco.images.append({"id": 1, "width": w, "height": h})
    
    model_repos = [
        ("omoured/YOLOv10-Document-Layout-Analysis", "yolov10n_best.pt"),
        ("omoured/YOLOv10-Document-Layout-Analysis", "yolov10s_best.pt"),
    ]
    
    mdl = None
    for repo_id, filename in model_repos:
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=None
            )
            mdl = YOLO(model_path)
            break
        except Exception:
            continue
    
    if mdl is None:
        raise RuntimeError(f"Could not load any YOLO document layout model. Tried: {model_repos}")
    
    pil_img = Image.fromarray(image.astype(np.uint8))
    res = mdl(pil_img, verbose=False)[0]
    
    cat_map = {}
    ann_id = 1
    
    if res.boxes is not None and len(res.boxes) > 0:
        for box, cls in zip(res.boxes.xywh, res.boxes.cls):
            name = mdl.names[int(cls)]
            
            if name != "Picture":
                continue
            
            if name not in cat_map:
                cid = len(coco.categories) + 1
                cat_map[name] = cid
                coco.categories.append({"id": cid, "name": name})
            
            cx, cy, bw, bh = box.tolist()
            x_min = max(0, int(cx - bw / 2))
            y_min = max(0, int(cy - bh / 2))
            bbox_width = min(int(bw), w - x_min)
            bbox_height = min(int(bh), h - y_min)
            
            coco.annotations.append({
                "id": ann_id,
                "image_id": 1,
                "category_id": cat_map[name],
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            })
            ann_id += 1
    
    return SegmentedImage(image=image, segmentation=coco)

