from dataclasses import dataclass
import numpy as np

from .format import SegmentationFormat


@dataclass
class SegmentedImage:
    image: np.ndarray
    segmentation: SegmentationFormat

