from pathlib import Path
import io
import numpy as np
import fitz
from PIL import Image


def load_image_pdf(path: Path) -> np.ndarray:
    pdf_document = fitz.open(path)
    if len(pdf_document) == 0:
        raise ValueError(f"PDF file {path} contains no pages")
    
    first_page = pdf_document[0]
    pixmap = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))
    
    img_bytes = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    img_array = np.array(img)
    pdf_document.close()
    
    return img_array

def load_image_png(path: Path) -> np.ndarray:
    img = Image.open(path)
    
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    return np.array(img)


def load_image(path: Path) -> np.ndarray:
    match path.suffix.lower():
        case ".pdf":
            return load_image_pdf(path)
        case ".png":
            return load_image_png(path)
        case _:
            raise ValueError(f"Unsupported image format: {path.suffix}")
