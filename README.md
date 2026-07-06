<<<<<<< HEAD
# 🏥 Medical Lesion Detection & VQA API

Hệ thống API hỗ trợ chẩn đoán hình ảnh y tế đa phương thức (Multi-modal AI). Hệ thống tự động phân tích ảnh X-quang ngực để phát hiện tổn thương, khoanh vùng vị trí, sinh câu mô tả (Captioning) và hỗ trợ trả lời câu hỏi chuyên sâu (VQA) dựa trên bệnh lý phát hiện được.
 
## ✨ Tính năng nổi bật

- **Detection & Heatmap:** Nhận diện các bệnh lý về phổi và vẽ bản đồ nhiệt (Grad-CAM) khoanh vùng tổn thương (TorchXRayVision - DenseNet121).
- **Image Captioning:** Tự động sinh câu mô tả y khoa chi tiết cho bức ảnh (Encoder ResNet50 + Decoder LSTM).
- **Medical VQA:** Trợ lý ảo AI trả lời các câu hỏi y khoa của người dùng dựa trên hình ảnh và bệnh lý đã chẩn đoán (Tích hợp Gemini 3 Flash Preview).
- **Tối ưu hiệu năng (High Performance):**
    - Tích hợp bộ nhớ đệm **Cache (Hash MD5)** giúp phản hồi tức thì với các ảnh đã từng được phân tích.
    - Sử dụng **Asynchronous (Bất đồng bộ)** để chạy song song luồng tính toán Captioning và luồng gọi mạng VQA, giúp giảm tối đa độ trễ.

---

## 📂 Cấu trúc thư mục (Directory Structure)

## ⚙️ Cài đặt môi trường (Installation)

**Bước 1: Clone dự án và di chuyển vào thư mục**

```bash
git clone <link-repo-cua-ban>
cd vào thư mục của mình
```

**Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)**
_Đối với Windows (PowerShell):_

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

_Nếu dự án của bạn đang dùng `.venv` thì thay bằng:_

```powershell
.\.venv\Scripts\Activate.ps1
```

_Đối với macOS/Linux:_

```bash
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện phụ thuộc**

```bash
pip install -r requirements.txt
```

_(Lưu ý: Hệ thống yêu cầu cài đặt `google-genai` để chạy nhánh VQA và `torch` phiên bản phù hợp với thiết bị của bạn)._

**Bước 4: Cấu hình API Key**
Đảm bảo bạn đã có Google Gemini API Key. Bạn có thể cấu hình Key này trực tiếp trong file `vqa_service.py` hoặc thiết lập biến môi trường.

---

## 🚀 Hướng dẫn chạy (How to Run)

### Khởi động nhanh

Nếu bạn chỉ muốn mở project nhanh nhất có thể, vì backend đã serve luôn frontend ở root, chỉ cần chạy FastAPI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Chế độ phát triển

Nếu muốn tự reload khi sửa code:

```bash
uvicorn backend.app:app --reload
```

_(Ghi chú: Cờ `--reload` giúp server tự động cập nhật khi bạn có thay đổi trong code)._

Bạn cũng có thể dùng đầy đủ lệnh PowerShell nếu đang làm việc trên Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Sau khi Terminal báo `Uvicorn running on http://127.0.0.1:8000`, hệ thống đã sẵn sàng!

---

## 🧪 Kiểm thử API (Testing with Swagger UI)

FastAPI tự động sinh tài liệu và giao diện kiểm thử chuẩn OpenAPI.

1. Mở trình duyệt và truy cập: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
2. Tìm đến endpoint `POST /analyze`.
3. Bấm **"Try it out"**.
4. Chọn một ảnh X-quang ngực để tải lên ở trường `file`.
5. (Tùy chọn) Nhập câu hỏi vào trường `question` (VD: _"Tôi có bị tràn dịch màng phổi không?"_).
6. Bấm **Execute** và xem kết quả JSON trả về.

Bạn có thể tùy chỉnh lại một vài chỗ như tên tác giả, link Github (nếu có) để file `README.md` mang đậm dấu ấn cá nhân của nhóm bạn nhé!
=======
# 🏥 Medical Lesion Detection & VQA API

Hệ thống API hỗ trợ chẩn đoán hình ảnh y tế đa phương thức (Multi-modal AI). Hệ thống tự động phân tích ảnh X-quang ngực để phát hiện tổn thương, khoanh vùng vị trí, sinh câu mô tả (Captioning) và hỗ trợ trả lời câu hỏi chuyên sâu (VQA) dựa trên bệnh lý phát hiện được.
 
## ✨ Tính năng nổi bật

- **Detection & Heatmap:** Nhận diện các bệnh lý về phổi và vẽ bản đồ nhiệt (Grad-CAM) khoanh vùng tổn thương (TorchXRayVision - DenseNet121).
- **Image Captioning:** Tự động sinh câu mô tả y khoa chi tiết cho bức ảnh (Encoder ResNet50 + Decoder LSTM).
- **Medical VQA:** Trợ lý ảo AI trả lời các câu hỏi y khoa của người dùng dựa trên hình ảnh và bệnh lý đã chẩn đoán (Tích hợp Gemini 3 Flash Preview).
- **Tối ưu hiệu năng (High Performance):**
    - Tích hợp bộ nhớ đệm **Cache (Hash MD5)** giúp phản hồi tức thì với các ảnh đã từng được phân tích.
    - Sử dụng **Asynchronous (Bất đồng bộ)** để chạy song song luồng tính toán Captioning và luồng gọi mạng VQA, giúp giảm tối đa độ trễ.

---

## 📂 Cấu trúc thư mục (Directory Structure)

## ⚙️ Cài đặt môi trường (Installation)

**Bước 1: Clone dự án và di chuyển vào thư mục**

```bash
git clone <link-repo-cua-ban>
cd vào thư mục của mình
```

**Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)**
_Đối với Windows (PowerShell):_

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

_Nếu dự án của bạn đang dùng `.venv` thì thay bằng:_

```powershell
.\.venv\Scripts\Activate.ps1
```

_Đối với macOS/Linux:_

```bash
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện phụ thuộc**

```bash
pip install -r requirements.txt
```

_(Lưu ý: Hệ thống yêu cầu cài đặt `google-genai` để chạy nhánh VQA và `torch` phiên bản phù hợp với thiết bị của bạn)._

**Bước 4: Cấu hình API Key**
Đảm bảo bạn đã có Google Gemini API Key. Bạn có thể cấu hình Key này trực tiếp trong file `vqa_service.py` hoặc thiết lập biến môi trường.

---

## 🚀 Hướng dẫn chạy (How to Run)

### Khởi động nhanh

Nếu bạn chỉ muốn mở project nhanh nhất có thể, vì backend đã serve luôn frontend ở root, chỉ cần chạy FastAPI:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Chế độ phát triển

Nếu muốn tự reload khi sửa code:

```bash
uvicorn backend.app:app --reload
```

_(Ghi chú: Cờ `--reload` giúp server tự động cập nhật khi bạn có thay đổi trong code)._

Bạn cũng có thể dùng đầy đủ lệnh PowerShell nếu đang làm việc trên Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Sau khi Terminal báo `Uvicorn running on http://127.0.0.1:8000`, hệ thống đã sẵn sàng!

---

## 🧪 Kiểm thử API (Testing with Swagger UI)

FastAPI tự động sinh tài liệu và giao diện kiểm thử chuẩn OpenAPI.

1. Mở trình duyệt và truy cập: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
2. Tìm đến endpoint `POST /analyze`.
3. Bấm **"Try it out"**.
4. Chọn một ảnh X-quang ngực để tải lên ở trường `file`.
5. (Tùy chọn) Nhập câu hỏi vào trường `question` (VD: _"Tôi có bị tràn dịch màng phổi không?"_).
6. Bấm **Execute** và xem kết quả JSON trả về.

Bạn có thể tùy chỉnh lại một vài chỗ như tên tác giả, link Github (nếu có) để file `README.md` mang đậm dấu ấn cá nhân của nhóm bạn nhé!
>>>>>>> 2c95cd330fa2d75770d21db6b8913872a885ec9e
