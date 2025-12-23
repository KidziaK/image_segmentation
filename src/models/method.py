from enum import Enum, auto


class SegmentationMethod(Enum):
    ImageDetectionCV = auto()
    YOLO = auto()

