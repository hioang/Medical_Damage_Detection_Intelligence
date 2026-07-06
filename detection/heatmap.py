# utils/heatmap.py
import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .load_model import preprocess_image_torchxray
from .detection import get_target_layers


def create_lung_mask(image_gray: np.ndarray) -> np.ndarray:
    """
    Lung mask heuristic tốt hơn bản cũ:
    - Loại vùng ngoài trung tâm
    - Dùng Otsu threshold
    - Chỉ giữ 2 contour lớn giống 2 phổi
    Lưu ý: tốt nhất vẫn là dùng model lung segmentation riêng.
    """
    h, w = image_gray.shape

    img = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # CLAHE giúp tăng tương phản vùng phổi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    # Với X-ray, vùng phổi thường tối hơn, dùng THRESH_BINARY_INV
    _, binary = cv2.threshold(
        img_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Chỉ giữ vùng giải phẫu hợp lý, tránh vai/cánh tay/cạnh ảnh
    roi = np.zeros_like(binary)
    x1, x2 = int(0.12 * w), int(0.88 * w)
    y1, y2 = int(0.12 * h), int(0.88 * h)
    roi[y1:y2, x1:x2] = 255
    binary = cv2.bitwise_and(binary, roi)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    img_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.015 * img_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Loại contour quá thấp/cao/quá sát biên
        if y < 0.08 * h or y + bh > 0.95 * h:
            continue
        if x < 0.05 * w or x + bw > 0.95 * w:
            continue

        aspect = bh / (bw + 1e-6)
        if aspect < 0.8 or aspect > 4.5:
            continue

        candidates.append((area, cnt))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:2]

    mask = np.zeros_like(image_gray, dtype=np.uint8)

    for _, cnt in candidates:
        cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Fallback nếu heuristic thất bại
    if cv2.countNonZero(mask) < 0.03 * img_area:
        mask = np.zeros_like(image_gray, dtype=np.uint8)
        mask[int(0.15*h):int(0.85*h), int(0.15*w):int(0.85*w)] = 255

    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def keep_contours_inside_lung(
    mask: np.ndarray,
    lung_mask: np.ndarray,
    min_area_ratio: float = 0.002,
    min_lung_overlap: float = 0.70,
) -> np.ndarray:
    h, w = mask.shape
    img_area = h * w

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * img_area:
            continue

        cnt_mask = np.zeros_like(mask)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

        overlap = cv2.countNonZero(cv2.bitwise_and(cnt_mask, lung_mask))
        cnt_pixels = cv2.countNonZero(cnt_mask)
        overlap_ratio = overlap / (cnt_pixels + 1e-6)

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Loại vùng sát biên ảnh
        if x <= 5 or y <= 5 or x + bw >= w - 5 or y + bh >= h - 5:
            continue

        # Chỉ giữ vùng thật sự nằm trong phổi
        if overlap_ratio >= min_lung_overlap:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)

    return filtered


def generate_focused_heatmap(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    target_class_index: int,
    layer_name: str = "denseblock3",
    heatmap_percentile: float = 90,
    blur_kernel: int = 5,
    apply_closing: bool = True,
    use_lung_mask: bool = True,
    min_area_ratio: float = 0.002,
    min_lung_overlap: float = 0.70,
):
    original = cv2.imread(image_path)

    if original is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    h, w = original_gray.shape

    img_tensor = preprocess_image_torchxray(image_path).to(device)

    target_layers = get_target_layers(model, layer_name=layer_name)

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_index)]

    grayscale_cam = cam(
        input_tensor=img_tensor,
        targets=targets
    )[0, :]

    heatmap_raw = cv2.resize(grayscale_cam, (w, h))
    heatmap_raw = np.maximum(heatmap_raw, 0)

    if heatmap_raw.max() > 0:
        heatmap_raw = heatmap_raw / heatmap_raw.max()

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    heatmap_blurred = cv2.GaussianBlur(
        heatmap_raw, (blur_kernel, blur_kernel), 0
    )

    if use_lung_mask:
        lung_mask = create_lung_mask(original_gray)
        lung_bin = (lung_mask > 0).astype(np.float32)

        # FIX QUAN TRỌNG:
        # Nhân heatmap với lung mask trước khi threshold.
        heatmap_blurred = heatmap_blurred * lung_bin
    else:
        lung_mask = np.ones_like(original_gray, dtype=np.uint8) * 255
        lung_bin = np.ones_like(original_gray, dtype=np.float32)

    inside_values = heatmap_blurred[lung_bin > 0]

    if inside_values.size == 0 or inside_values.max() <= 0:
        mask = np.zeros_like(original_gray, dtype=np.uint8)
    else:
        # FIX QUAN TRỌNG:
        # Dùng percentile thay vì ngưỡng cố định.
        threshold_value = np.percentile(
            inside_values[inside_values > 0],
            heatmap_percentile
        )

        mask = (
            (heatmap_blurred >= threshold_value) &
            (lung_bin > 0)
        ).astype(np.uint8) * 255

    if apply_closing:
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if use_lung_mask:
        mask = keep_contours_inside_lung(
            mask=mask,
            lung_mask=lung_mask,
            min_area_ratio=min_area_ratio,
            min_lung_overlap=min_lung_overlap,
        )

    heatmap_color = cv2.applyColorMap(
        (heatmap_blurred * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = original_rgb.copy()

    if cv2.countNonZero(mask) > 0:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        alpha = 0.55

        overlay = (
            overlay * (1 - mask_3ch) +
            (alpha * heatmap_color + (1 - alpha) * overlay) * mask_3ch
        ).astype(np.uint8)

    bbox_image = overlay.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * img_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        cv2.rectangle(
            bbox_image,
            (x, y),
            (x + bw, y + bh),
            (0, 255, 0),
            2
        )

        cv2.putText(
            bbox_image,
            "Lesion",
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return overlay, bbox_image, mask