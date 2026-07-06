import torch
import os
from PIL import Image
from torchvision import transforms
import pandas as pd

from model import CaptionModel
from vocab import Vocabulary
 
# load vocab
if not os.path.exists("vocab.pth"):
    raise FileNotFoundError("Vui lòng chạy train.py trước để tạo file vocab.pth")
# Cần weights_only=False để load class Vocabulary tùy chỉnh trên PyTorch mới
vocab = torch.load("vocab.pth", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptionModel(256, 512, len(vocab.itos)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def generate_caption(image):
    result = ["<start>"]
 
    for _ in range(50): # Tăng độ dài tối đa nếu cần
        inputs = torch.tensor([vocab.stoi[w] for w in result]).unsqueeze(0).to(device)
        outputs = model(image, inputs)

        # Do model nối feature (ảnh) vào đầu sequence: [Feature, <start>, word1, ...]
        # p_feature (index 0) -> dự đoán word1
        # p_start   (index 1) -> dự đoán word2
        # Chỉ số đúng để lấy từ tiếp theo luôn là len(result) - 1
        next_word = outputs.argmax(2)[0, len(result) - 1].item()
        word = vocab.itos[next_word]

        result.append(word)

        if word == "<end>":
            break

    # Loại bỏ các token điều khiển khi in kết quả
    final_caption = [w for w in result if w not in ["<start>", "<end>", "<pad>"]]
    return " ".join(final_caption)
 
<<<<<<< HEAD
test_image_path = "images/CXR3468_IM-1684-0001-0001.png"
=======
test_image_path = "images/CXM-1324-2001.png"
>>>>>>> 2c95cd330fa2d75770d21db6b8913872a885ec9e
img = Image.open(test_image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

print(generate_caption(img))