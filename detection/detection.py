# utils/detection.py
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any

from .load_model import preprocess_image_torchxray, get_severity, IGNORE_LABELS

logger = logging.getLogger(__name__)

USE_OP_THRESH = True
USE_TTA = True
TEMPERATURE = 1.15
DEFAULT_THRESHOLD = 0.55

PER_PATHOLOGY_RULES = {
    "Pneumothorax": 0.60,
    "Nodule": 0.70,
    "Mass": 0.70,
    "Fracture": 0.70,
    "Hernia": 0.70,
    "Cardiomegaly": 0.60,
    "Effusion": 0.55,
    "Infiltration": 0.60,
    "Consolidation": 0.60,
    "Edema": 0.55,
    "Emphysema": 0.60,
    "Fibrosis": 0.60,
    "Pleural_Thickening": 0.65,
    "Lung Opacity": 0.60,
    "Atelectasis": 0.55,
    "Pneumonia": 0.60,
    "Lung Lesion": 0.65,
    "Enlarged Cardiomediastinum": 0.60,
}


def detect(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    use_operating_threshold: bool = USE_OP_THRESH,
    use_tta: bool = USE_TTA,
    use_temperature_scaling: bool = True,
    use_per_pathology_rules: bool = True
) -> Dict[str, Any]:

    if not hasattr(model, "pathologies"):
        raise AttributeError("Model thiếu thuộc tính 'pathologies'")

    img_tensor = preprocess_image_torchxray(image_path).to(device)

    feature_maps = []
    handle = None

    if hasattr(model, "features"):
        def hook_fn(module, inp, out):
            feature_maps.append(out.detach().cpu())
        handle = model.features.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model(img_tensor)[0]
            probs = torch.sigmoid(out) if getattr(model, "apply_sigmoid", False) is False else out

        if use_tta:
            flipped = torch.flip(img_tensor, dims=[-1])
            with torch.no_grad():
                out_flip = model(flipped)[0]
                prob_flip = torch.sigmoid(out_flip) if getattr(model, "apply_sigmoid", False) is False else out_flip
            probs = (probs + prob_flip) / 2.0

        if use_temperature_scaling:
            logits = torch.logit(probs, eps=1e-7)
            probs = torch.sigmoid(logits / TEMPERATURE)

    finally:
        if handle:
            handle.remove()

    scores = probs.detach().cpu().numpy()

    if use_operating_threshold:
        op_threshs = getattr(model, "op_threshs", None)
        if op_threshs is not None:
            if torch.is_tensor(op_threshs):
                op_threshs = op_threshs.cpu().numpy()
            base_thresholds = [
                float(th) if th is not None else threshold
                for th in op_threshs
            ]
        else:
            base_thresholds = [threshold] * len(scores)
    else:
        base_thresholds = [threshold] * len(scores)

    all_results = []

    for idx, disease in enumerate(model.pathologies):
        if disease in IGNORE_LABELS:
            continue

        conf = float(scores[idx])
        th = base_thresholds[idx]

        if use_per_pathology_rules and disease in PER_PATHOLOGY_RULES:
            th = max(th, PER_PATHOLOGY_RULES[disease])

        all_results.append({
            "name": disease,
            "confidence": conf,
            "class_index": idx,
            "threshold": th,
        })

    all_results.sort(key=lambda x: x["confidence"], reverse=True)

    detected = [
        r for r in all_results
        if r["confidence"] >= r["threshold"]
    ]

    has_lesion = len(detected) > 0
    severity = get_severity(detected[0]["confidence"]) if has_lesion else "normal"

    # FIX QUAN TRỌNG:
    # Chỉ lấy class đã vượt ngưỡng, không lấy class score cao nhất một cách mù quáng.
    top_class_index = detected[0]["class_index"] if detected else None
    top_class_name = detected[0]["name"] if detected else None

    feature_map_out = feature_maps[0] if feature_maps else None

    pathologies_out = [
        {
            "name": r["name"],
            "confidence": r["confidence"],
            "class_index": r["class_index"],
            "threshold": r["threshold"],
        }
        for r in detected[:10]
    ]

    return {
        "has_lesion": has_lesion,
        "pathologies": pathologies_out,
        "severity": severity,
        "feature_maps": feature_map_out,
        "top_class_index": top_class_index,
        "top_class_name": top_class_name,
        "all_results": all_results[:10],
    }


def get_target_layers(model: nn.Module, layer_name: str = "denseblock3") -> List[nn.Module]:
    if not hasattr(model, "features"):
        raise RuntimeError("Model không có 'features'")

    features = model.features

    if layer_name == "denseblock4":
        block = features.denseblock4
    elif layer_name == "denseblock3":
        block = features.denseblock3
    elif layer_name == "denseblock2":
        block = features.denseblock2
    else:
        raise ValueError(f"Layer {layer_name} không hợp lệ")

    for name, module in reversed(list(block.named_modules())):
        if isinstance(module, nn.Conv2d):
            return [module]

    raise RuntimeError(f"Không tìm thấy Conv2d trong {layer_name}")