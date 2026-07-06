# utils/load_model.py
import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import numpy as np
import skimage.io
from PIL import Image

# ========== CẤU HÌNH ==========
WEIGHTS_NAME = "densenet121-res224-all"
IGNORE_LABELS = {"", "Support Devices"}

# ========== TIỀN XỬ LÝ ẢNH ==========
def preprocess_image_torchxray(image_path: str) -> torch.Tensor:
    """Đọc ảnh, chuẩn hóa về tensor [1, 1, 224, 224] theo chuẩn torchxrayvision."""
    try:
        img = skimage.io.imread(image_path)
    except Exception:
        img = np.array(Image.open(image_path).convert("RGB"))

    # Xử lý kênh màu
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.mean(axis=2)          # chuyển grayscale
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    if img.ndim != 2:
        raise ValueError(f"Shape không hợp lệ: {img.shape}")

    # Xác định maxval
    if img.dtype == np.uint8:
        maxval = 255.0
    elif img.dtype == np.uint16:
        maxval = 65535.0
    else:
        img = img.astype(np.float32)
        maxval = img.max()
        if maxval <= 1.0:
            maxval = 1.0
        elif maxval <= 255.0:
            maxval = 255.0
        else:
            maxval = 65535.0

    img = xrv.datasets.normalize(img, maxval)   # chuẩn hóa về [0,1]
    img = img[None, ...]                        # thêm chiều channel

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    return torch.from_numpy(img).float().unsqueeze(0)   # [1,1,224,224]

# ========== LOAD MODEL ==========
def load_model(device: torch.device):
    """Load model torchxrayvision, tự động xử lý sigmoid."""
    model = xrv.models.DenseNet(weights=WEIGHTS_NAME)
    if getattr(model, "op_threshs", None) is None:
        model.apply_sigmoid = True
    model = model.to(device)
    model.eval()
    return model

# ========== SEVERITY ==========
def get_severity(score: float) -> str:
    """Chuyển score thành mức độ (chỉ mang tính tham khảo)."""
    if score < 0.5:
        return "normal"
    if score < 0.65:
        return "low"
    if score < 0.80:
        return "moderate"
    return "high"