# Hệ thống mô tả ảnh y tế (Image Captioning)

## 1. Giới thiệu

Dự án này triển khai một hệ thống mô tả ảnh tự động (Image Captioning) sử dụng kiến trúc Encoder-Decoder. Mục tiêu là sinh ra các câu mô tả ngắn gọn và chính xác cho các hình ảnh y tế, đặc biệt là ảnh X-quang.

Kiến trúc chính bao gồm:
*   **Encoder:** Sử dụng mô hình ResNet50 (đã được huấn luyện trước trên ImageNet) để trích xuất các đặc trưng thị giác từ hình ảnh.
*   **Decoder:** Sử dụng mạng LSTM để sinh ra chuỗi từ (caption) dựa trên các đặc trưng được trích xuất từ Encoder.

## 1. Cấu trúc thư mục

```text
.
├── images/                 # Chứa các file ảnh (ví dụ: CXR647_IM-2225-2001.png)
├── caption.csv             # File CSV chứa ánh xạ giữa tên ảnh và caption
├── train.py                # Script huấn luyện mô hình
├── test.py                 # Script kiểm thử/sinh caption cho ảnh mới
├── model.py                # Định nghĩa kiến trúc EncoderCNN, DecoderRNN và CaptionModel
├── dataset.py              # Định nghĩa lớp CaptionDataset để tải dữ liệu
├── vocab.py                # Định nghĩa lớp Vocabulary để xử lý từ vựng
├── utils.py                # Chứa các hàm tiện ích (ví dụ: collate_fn)
├── vocab.pth               # (Được tạo sau khi huấn luyện) File lưu trữ từ vựng
├── model.pth               # (Được tạo sau khi huấn luyện) File lưu trữ trọng số mô hình
└── README_VI.md            # File hướng dẫn này
```

## 2. Cách điền captions.csv

Mỗi dòng gồm:

```csv
image_id,caption
img001.png,không phát hiện tổn thương rõ ràng
img002.png,phát hiện vùng mờ bất thường ở phổi phải
```

Lưu ý:
- `image_id` phải trùng với tên file ảnh trong thư mục `images/`.
- Caption nên viết chữ thường, ngắn, rõ ý.
- Nếu một ca bệnh có nhiều ảnh, có thể ghi: `img001_front.png,img001_lateral.png` trong cột `image_id`.

## 3. Chạy tạo dataset

```bash
python prepare_biocaption_dataset.py --csv captions.csv --images images --out output
```

Sau khi chạy, thư mục `output/` sẽ có:

```text
train_images.tsv
val_images.tsv
test_images.tsv
vocab.json
dataset_summary.json
missing_images.txt
```

## 4. Dùng với bioCaption

Ví dụ baseline:

```python
from bioCaption.models.captionModels.baselines import Baselines

baselines = Baselines(
    'output/train_images.tsv',
    'output/test_images.tsv',
    'images/',
    'results'
)
baselines.most_frequent_word_in_captions()
```

## 5. Format TSV đúng của bioCaption

```tsv
image_name.png<TAB>caption
image_front.png,image_lateral.png<TAB>caption
```

Giữa tên ảnh và caption là dấu TAB, không phải dấu phẩy.
