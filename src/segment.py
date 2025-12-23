from pathlib import Path

from .models import SegmentationMethod, SegmentationFormat, COCO, SegmentedImage
from .models.image_loader import load_image
from .layoutlmv3 import segment_image_layoutlmv3


def segment_image(path: Path, method: SegmentationMethod = SegmentationMethod.LayoutLMv3) -> SegmentedImage:
    image = load_image(path)

    match method:
        case SegmentationMethod.LayoutLMv3:
            return segment_image_layoutlmv3(image)
        case _:
            raise ValueError(f"Unsupported segmentation method: {method}")