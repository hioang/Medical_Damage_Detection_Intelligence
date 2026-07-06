<<<<<<< HEAD
import torch
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from model import CaptionModel
from vocab import Vocabulary

def evaluate_model(csv_test_path, image_folder, model_path, vocab_path, device):
    # 1. Load Vocab & Model
    vocab = torch.load(vocab_path, weights_only=False)
    model = CaptionModel(256, 512, len(vocab.itos)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Test Data
    data = []
    with open(csv_test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                img_id = parts[0].strip().strip('"')
                caption = parts[1].strip().strip('"').lower()
                data.append([img_id, caption])

    df_test = pd.DataFrame(data, columns=["image_id", "caption"])
    
    results = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    print(f"Đang đánh giá trên {len(df_test)} mẫu thử nghiệm...")
    
    with torch.no_grad():
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
            img_name = row['image_id']
            reference_caption = str(row['caption']).lower().split()
            
            img_path = os.path.join(image_folder, img_name)
            if not os.path.exists(img_path):
                continue

            # Sinh caption từ model
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            
            generated_tokens = ["<start>"]
            for _ in range(50):
                inputs = torch.tensor([vocab.stoi[w] for w in generated_tokens]).unsqueeze(0).to(device)
                outputs = model(img, inputs)
                next_word_idx = outputs.argmax(2)[0, len(generated_tokens) - 1].item()
                word = vocab.itos[next_word_idx]
                if word == "<end>": break
                generated_tokens.append(word)
            
            predicted_caption = [w for w in generated_tokens if w not in ["<start>", "<end>", "<pad>"]]
            
            # Sử dụng tokenizer của vocab để đồng bộ dữ liệu test và predict
            ref_tokens = vocab.tokenizer(row['caption'])
            pred_tokens = predicted_caption

            # Tính điểm
            bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            rouge_l = scorer.score(" ".join(reference_caption), " ".join(predicted_caption))['rougeL'].fmeasure

            results.append({
                'bleu1': bleu1,
                'bleu4': bleu4,
                'rougeL': rouge_l
            })

    # 3. Tổng hợp bảng điểm
    res_df = pd.DataFrame(results)
    summary = {
        "Metric": ["BLEU-1", "BLEU-4", "ROUGE-L"],
        "Average Score": [
            f"{res_df['bleu1'].mean():.4f}",
            f"{res_df['bleu4'].mean():.4f}",
            f"{res_df['rougeL'].mean():.4f}"
        ],
        "Min": [res_df['bleu1'].min(), res_df['bleu4'].min(), res_df['rougeL'].min()],
        "Max": [res_df['bleu1'].max(), res_df['bleu4'].max(), res_df['rougeL'].max()]
    }
    
    print("\n" + "="*30)
    print("BẢNG ĐIỂM ĐÁNH GIÁ MÔ HÌNH")
    print("="*30)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print("="*30)
    
    # Lưu kết quả ra file csv
    summary_df.to_csv("evaluation_results.csv", index=False)
    print("Đã lưu kết quả vào file evaluation_results.csv")

if __name__ == "__main__":
    # Sử dụng tập test riêng biệt để đánh giá khách quan
    CSV_PATH = "test_caption.csv" 
    IMG_DIR = "images"
    MODEL_PATH = "model.pth"
    VOCAB_PATH = "vocab.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
=======
import torch
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from model import CaptionModel
from vocab import Vocabulary

def evaluate_model(csv_test_path, image_folder, model_path, vocab_path, device):
    # 1. Load Vocab & Model
    vocab = torch.load(vocab_path, weights_only=False)
    model = CaptionModel(256, 512, len(vocab.itos)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load Test Data
    data = []
    with open(csv_test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                img_id = parts[0].strip().strip('"')
                caption = parts[1].strip().strip('"').lower()
                data.append([img_id, caption])

    df_test = pd.DataFrame(data, columns=["image_id", "caption"])
    
    results = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    print(f"Đang đánh giá trên {len(df_test)} mẫu thử nghiệm...")
    
    with torch.no_grad():
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
            img_name = row['image_id']
            reference_caption = str(row['caption']).lower().split()
            
            img_path = os.path.join(image_folder, img_name)
            if not os.path.exists(img_path):
                continue

            # Sinh caption từ model
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            
            generated_tokens = ["<start>"]
            for _ in range(50):
                inputs = torch.tensor([vocab.stoi[w] for w in generated_tokens]).unsqueeze(0).to(device)
                outputs = model(img, inputs)
                next_word_idx = outputs.argmax(2)[0, len(generated_tokens) - 1].item()
                word = vocab.itos[next_word_idx]
                if word == "<end>": break
                generated_tokens.append(word)
            
            predicted_caption = [w for w in generated_tokens if w not in ["<start>", "<end>", "<pad>"]]
            
            # Sử dụng tokenizer của vocab để đồng bộ dữ liệu test và predict
            ref_tokens = vocab.tokenizer(row['caption'])
            pred_tokens = predicted_caption

            # Tính điểm
            bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
            rouge_l = scorer.score(" ".join(reference_caption), " ".join(predicted_caption))['rougeL'].fmeasure

            results.append({
                'bleu1': bleu1,
                'bleu4': bleu4,
                'rougeL': rouge_l
            })

    # 3. Tổng hợp bảng điểm
    res_df = pd.DataFrame(results)
    summary = {
        "Metric": ["BLEU-1", "BLEU-4", "ROUGE-L"],
        "Average Score": [
            f"{res_df['bleu1'].mean():.4f}",
            f"{res_df['bleu4'].mean():.4f}",
            f"{res_df['rougeL'].mean():.4f}"
        ],
        "Min": [res_df['bleu1'].min(), res_df['bleu4'].min(), res_df['rougeL'].min()],
        "Max": [res_df['bleu1'].max(), res_df['bleu4'].max(), res_df['rougeL'].max()]
    }
    
    print("\n" + "="*30)
    print("BẢNG ĐIỂM ĐÁNH GIÁ MÔ HÌNH")
    print("="*30)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print("="*30)
    
    # Lưu kết quả ra file csv
    summary_df.to_csv("evaluation_results.csv", index=False)
    print("Đã lưu kết quả vào file evaluation_results.csv")

if __name__ == "__main__":
    # Sử dụng tập test riêng biệt để đánh giá khách quan
    CSV_PATH = "test_caption.csv" 
    IMG_DIR = "images"
    MODEL_PATH = "model.pth"
    VOCAB_PATH = "vocab.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> 2c95cd330fa2d75770d21db6b8913872a885ec9e
    evaluate_model(CSV_PATH, IMG_DIR, MODEL_PATH, VOCAB_PATH, device)