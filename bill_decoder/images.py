"""Image variants in memory only. Original files are never written."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_VERSION = '1'

@dataclass(frozen=True)
class ImageOptions:
    rotate: int = 0
    contrast: bool = False
    detail_crops: bool = False
    crop: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.rotate not in (0, 90, 180, 270):
            raise ValueError('Rotation must be 0/90/180/270 counterclockwise degrees')
        if self.crop and (len(self.crop) != 4 or not all(0 <= x <= 1 for x in self.crop) or self.crop[0] >= self.crop[2] or self.crop[1] >= self.crop[3]):
            raise ValueError('Crop must be normalized x0,y0,x1,y1')


def load_image(path: Path, options: ImageOptions):
    with Image.open(path) as source:
        if source.format not in ('PNG', 'JPEG'):
            raise ValueError('Only PNG/JPG bill images are supported')
        image = ImageOps.exif_transpose(source).convert('RGB')
    if options.rotate:
        image = image.rotate(options.rotate, expand=True)
    return image


def png_bytes(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def region(image, box):
    w, h = image.size
    return image.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))


def variants(image, options):
    views = [('full', image)]
    if options.contrast:
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        views.append(('contrast_full', Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))))
    if options.detail_crops and image.width < 1100:
        for name, box in [('upper', (0, 0, 1, .62)), ('lower', (0, .38, 1, 1))]:
            crop = region(image, box)
            views.append((name + ' crop=' + str(box), crop.resize((crop.width*2, crop.height*2), Image.Resampling.LANCZOS)))
    if options.crop:
        views.append(('target crop=' + str(options.crop), region(image, options.crop)))
    return [(name, png_bytes(view)) for name, view in views]
