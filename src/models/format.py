from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import json


class SegmentationFormat(ABC):
    @abstractmethod
    def export(self, path: Path | str) -> None:
        pass


@dataclass
class COCO(SegmentationFormat):
    images: list[dict[str, int | str]] = field(default_factory=list)
    annotations: list[dict[str, int | float]] = field(default_factory=list)
    categories: list[dict[str, int | str]] = field(default_factory=list)
    
    def export(self, path: Path | str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        coco_data = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        
        with path_obj.open("w") as f:
            json.dump(coco_data, f, indent=2)

