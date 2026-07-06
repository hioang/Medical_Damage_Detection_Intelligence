<<<<<<< HEAD
import torch
import os
from PIL import Image
from torchvision import transforms

# Sử dụng import tương đối hoặc đảm bảo path đã được thêm ở app.py
try:
    # Khi chạy từ app.py (thư mục captioning đã được add vào sys.path)
    from model import CaptionModel
    from vocab import Vocabulary
except ImportError:
    # Khi chạy trực tiếp hoặc cấu trúc khác
    from captioning.model import CaptionModel
    from captioning.vocab import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến lưu trữ model và vocab (Singleton) để không phải load lại mỗi khi gọi API
_model = None
_vocab = None
 
def _load_resources():
    global _model, _vocab
    if _model is None or _vocab is None:
        # Xác định đường dẫn file weights và vocab
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(current_dir, "vocab.pth")
        model_path = os.path.join(current_dir, "model.pth")

        # Nếu không tìm thấy ở thư mục local, thử tìm ở thư mục project root
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join("captioning", "vocab.pth")
            model_path = os.path.join("captioning", "model.pth")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Không tìm thấy file vocab.pth tại {vocab_path}. Vui lòng chạy train.py trước.")
            
        _vocab = torch.load(vocab_path, weights_only=False)
        _model = CaptionModel(256, 512, len(_vocab.itos)).to(device)
        
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Cảnh báo: Không tìm thấy file trọng số model.pth tại {model_path}")
            
        _model.eval()
    return _model, _vocab

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_caption(image_path):
    """
    Hàm thực hiện inference để sinh ra câu mô tả từ đặc trưng ảnh.
    """
    try:
        model, vocab = _load_resources()
        
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        result = ["<start>"]
        with torch.no_grad():
            for _ in range(50):
                inputs = torch.tensor([vocab.stoi.get(w, vocab.stoi.get("<unk>", 0)) for w in result]).unsqueeze(0).to(device)
                outputs = model(img, inputs)

                # Lấy từ có xác suất cao nhất tại bước hiện tại
                next_word_idx = outputs.argmax(2)[0, len(result) - 1].item()
                word = vocab.itos[next_word_idx]


                if word == "<end>":
                    break
                result.append(word)

        # Loại bỏ các token kỹ thuật và các chuỗi giống mã ID (tránh hiện tượng hiện CXR2585...)
        final_caption = []
        for w in result:
            if w in ["<start>", "<end>", "<pad>", "<unk>"]:
                continue
            # Bỏ qua các từ chứa dấu gạch ngang và số (thường là mã ảnh bị lẫn vào vocab)
            if "-" in w and any(char.isdigit() for char in w):
                continue
            final_caption.append(w)
        
        # Xử lý để trả về một câu miêu tả hoàn chỉnh
        description = " ".join(final_caption).strip()
        return description.capitalize() + "." if description else "Không thể tạo mô tả chi tiết."
        
    except Exception as e:
        print(f"Error in predict_caption: {e}")
=======
import torch
import os
from PIL import Image
from torchvision import transforms

# Sử dụng import tương đối hoặc đảm bảo path đã được thêm ở app.py
try:
    # Khi chạy từ app.py (thư mục captioning đã được add vào sys.path)
    from model import CaptionModel
    from vocab import Vocabulary
except ImportError:
    # Khi chạy trực tiếp hoặc cấu trúc khác
    from captioning.model import CaptionModel
    from captioning.vocab import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến lưu trữ model và vocab (Singleton) để không phải load lại mỗi khi gọi API
_model = None
_vocab = None
 
def _load_resources():
    global _model, _vocab
    if _model is None or _vocab is None:
        # Xác định đường dẫn file weights và vocab
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vocab_path = os.path.join(current_dir, "vocab.pth")
        model_path = os.path.join(current_dir, "model.pth")

        # Nếu không tìm thấy ở thư mục local, thử tìm ở thư mục project root
        if not os.path.exists(vocab_path):
            vocab_path = os.path.join("captioning", "vocab.pth")
            model_path = os.path.join("captioning", "model.pth")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Không tìm thấy file vocab.pth tại {vocab_path}. Vui lòng chạy train.py trước.")
            
        _vocab = torch.load(vocab_path, weights_only=False)
        _model = CaptionModel(256, 512, len(_vocab.itos)).to(device)
        
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Cảnh báo: Không tìm thấy file trọng số model.pth tại {model_path}")
            
        _model.eval()
    return _model, _vocab

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_caption(image_path):
    """
    Hàm thực hiện inference để sinh ra câu mô tả từ đặc trưng ảnh.
    """
    try:
        model, vocab = _load_resources()
        
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        result = ["<start>"]
        with torch.no_grad():
            for _ in range(50):
                inputs = torch.tensor([vocab.stoi.get(w, vocab.stoi.get("<unk>", 0)) for w in result]).unsqueeze(0).to(device)
                outputs = model(img, inputs)

                # Lấy từ có xác suất cao nhất tại bước hiện tại
                next_word_idx = outputs.argmax(2)[0, len(result) - 1].item()
                word = vocab.itos[next_word_idx]


                if word == "<end>":
                    break
                result.append(word)

        # Loại bỏ các token kỹ thuật và các chuỗi giống mã ID (tránh hiện tượng hiện CXR2585...)
        final_caption = []
        for w in result:
            if w in ["<start>", "<end>", "<pad>", "<unk>"]:
                continue
            # Bỏ qua các từ chứa dấu gạch ngang và số (thường là mã ảnh bị lẫn vào vocab)
            if "-" in w and any(char.isdigit() for char in w):
                continue
            final_caption.append(w)
        
        # Xử lý để trả về một câu miêu tả hoàn chỉnh
        description = " ".join(final_caption).strip()
        return description.capitalize() + "." if description else "Không thể tạo mô tả chi tiết."
        
    except Exception as e:
        print(f"Error in predict_caption: {e}")
>>>>>>> 2c95cd330fa2d75770d21db6b8913872a885ec9e
        return "Không thể khởi tạo mô tả cho hình ảnh này."