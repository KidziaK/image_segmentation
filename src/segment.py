from pathlib import Path

from .models import SegmentationMethod, SegmentationFormat, COCO, SegmentedImage
from .models.image_loader import load_image
from .segmentation_open_cv import segment_image_open_cv
from .segmentation_yolo import segment_image_yolo


def segment_image(path: Path, method: SegmentationMethod = SegmentationMethod.ImageDetectionCV) -> SegmentedImage:
    image = load_image(path)

    match method:
        case SegmentationMethod.ImageDetectionCV:
            return segment_image_open_cv(image)
        case SegmentationMethod.YOLO:
            return segment_image_yolo(image)
        case _:
            raise ValueError(f"Unsupported segmentation method: {method}")