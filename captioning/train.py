import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from dataset import CaptionDataset
from vocab import Vocabulary
from model import CaptionModel
from utils import collate_fn

# ===== PATH =====
csv_file = "train_caption.csv"
image_folder = "images"

# ===== LOAD DATA =====
# Đọc file CSV thủ công để tránh lỗi dấu phẩy trong caption
data = []
with open(csv_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            img_id = parts[0].strip().strip('"')
            caption = parts[1].strip().strip('"').lower() # Chuyển về chữ thường
            data.append([img_id, caption])

df = pd.DataFrame(data, columns=["image_id", "caption"]).dropna()

# Lọc để đảm bảo vocab được xây dựng từ caption của các ảnh có thật
mask = df["image_id"].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))
df = df[mask].reset_index(drop=True)
print(f"Tìm thấy {len(df)} ảnh hợp lệ để huấn luyện.")

sentences = df["caption"].tolist()

# ===== VOCAB =====
vocab = Vocabulary()
vocab.build_vocab(sentences)
# Lưu vocab để dùng cho việc test, đảm bảo ID từ không bị thay đổi
torch.save(vocab, "vocab.pth")

# ===== DATASET =====
dataset = CaptionDataset(image_folder, csv_file, vocab)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# ===== MODEL =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptionModel(256, 512, len(vocab.itos)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ===== TRAIN =====
for epoch in range(100): # Tăng epoch lên ít nhất 100 để mô hình kịp học
    total_loss = 0
    for imgs, caps in loader:
        imgs, caps = imgs.to(device), caps.to(device)

        outputs = model(imgs, caps[:, :-1])

        # Decoder nhận (Features + Captions[:-1]), nên output tương ứng sẽ dự đoán từ Captions[1:]
        # outputs[:, :-1] sẽ khớp với targets caps[:, 1:]
        # caps[:, 1:] bỏ qua token <start>
        loss = criterion(
            outputs[:, :-1].reshape(-1, outputs.shape[2]),
            caps[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} Average Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pth")