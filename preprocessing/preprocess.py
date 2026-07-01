from pathlib import Path
from typing import Tuple, Union


import cv2
import numpy as np

try:
	import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tensor output
	torch = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(image_or_path: Union[str, Path, np.ndarray]) -> np.ndarray:
	if isinstance(image_or_path, (str, Path)):
		image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError(f"Could not read image from path: {image_or_path}")
		return image

	if isinstance(image_or_path, np.ndarray):
		if image_or_path.size == 0:
			raise ValueError("Input image array is empty")
		return image_or_path

	raise TypeError("image_or_path must be a file path or numpy array")


def preprocess_image(
	image_or_path: Union[str, Path, np.ndarray],
	output_size: Tuple[int, int] = (224, 224),
	return_tensor: bool = False,
) -> Union[np.ndarray, "torch.Tensor"]:
	image = _load_image(image_or_path)

	if image.ndim == 2:
		gray = image
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)

	# Improve local contrast for chest X-ray style inputs.
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(resized)

	rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
	normalized = rgb.astype(np.float32) / 255.0
	normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
	chw = np.transpose(normalized, (2, 0, 1)).astype(np.float32)

	if return_tensor:
		if torch is None:
			raise ModuleNotFoundError("PyTorch is not installed. Set return_tensor=False or install torch.")
		return torch.from_numpy(chw)
	return chw
