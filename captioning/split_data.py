<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(input_csv, train_out, test_out, test_size=0.2):
    # Đọc dữ liệu (giữ logic split giống train.py)
    data = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        header = f.readline()
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                data.append(parts)

    df = pd.DataFrame(data, columns=["image_id", "caption"])
    
    # Chia tập dữ liệu
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Đã chia dữ liệu: {len(train_df)} mẫu train, {len(test_df)} mẫu test.")

if __name__ == "__main__":
=======
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(input_csv, train_out, test_out, test_size=0.2):
    # Đọc dữ liệu (giữ logic split giống train.py)
    data = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        header = f.readline()
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                data.append(parts)

    df = pd.DataFrame(data, columns=["image_id", "caption"])
    
    # Chia tập dữ liệu
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Đã chia dữ liệu: {len(train_df)} mẫu train, {len(test_df)} mẫu test.")

if __name__ == "__main__":
>>>>>>> 2c95cd330fa2d75770d21db6b8913872a885ec9e
    split_csv("caption.csv", "train_caption.csv", "test_caption.csv")