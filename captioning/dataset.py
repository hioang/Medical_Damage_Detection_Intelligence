import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CaptionDataset(Dataset):
    def __init__(self, root_dir, csv_file, vocab):
        self.root_dir = root_dir
        
        # Đọc file CSV thủ công để tránh lỗi dấu phẩy trong caption
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Bỏ qua header
            for line in lines:
                parts = line.strip().split(',', 1) # Chỉ split ở dấu phẩy đầu tiên
                if len(parts) == 2:
                    img_id = parts[0].strip().strip('"')
                    caption = parts[1].strip().strip('"').lower() # Lowercase đồng bộ
                    data.append([img_id, caption])
        
        df = pd.DataFrame(data, columns=["image_id", "caption"]).dropna()
        
        # Chỉ giữ lại các dòng mà file ảnh thực sự tồn tại
        mask = df["image_id"].apply(lambda x: os.path.exists(os.path.join(root_dir, x)))
        self.df = df[mask].reset_index(drop=True)
        
        self.vocab = vocab

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        caption = self.df.iloc[idx]["caption"]

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        numericalized = [self.vocab.stoi["<start>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized)