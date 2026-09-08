"""
BinX Tech AI & Machine Learning Internship — Sprint 3
Day 3 Export: Reusable OpenCV Preprocessing Engine
"""
from pathlib import Path
import numpy as np
import cv2

def preprocess_image_cv(image_input, target_size=(224, 224), normalization_mode='mobilenet', keep_aspect_ratio=False):
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Corrupt or invalid image file: {img_path}")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise TypeError(f"Invalid input type: {type(image_input)}")

    target_h, target_w = target_size
    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    elif img_bgr.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img_rgb.shape[:2]
    interp = cv2.INTER_AREA if (orig_h > target_h or orig_w > target_w) else cv2.INTER_LINEAR
    resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=interp)
    img_float = resized.astype(np.float32)

    if normalization_mode == 'standard':
        return img_float / 255.0
    elif normalization_mode == 'mobilenet':
        return (img_float / 127.5) - 1.0
    elif normalization_mode == 'none':
        return img_float
    else:
        raise ValueError(f"Unknown mode: {normalization_mode}")
