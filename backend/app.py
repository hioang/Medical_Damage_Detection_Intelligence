
import os
import cv2
import asyncio
import torch
import sys
import hashlib
import shutil
import base64
import io
import traceback
from io import BytesIO

# --- THIẾT LẬP ĐƯỜNG DẪN HỆ THỐNG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Đảm bảo thư mục captioning ưu tiên hàng đầu trong path để tránh import nhầm file inference ở root
CAPTIONING_PATH = os.path.join(PROJECT_ROOT, "captioning")
if CAPTIONING_PATH not in sys.path:
    sys.path.insert(0, CAPTIONING_PATH)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from vqa.vqa_service import answer_question
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw, ImageFont 

# Import Nhánh A
from detection.load_model import load_model
from detection.detection import detect
from detection.heatmap import generate_focused_heatmap

# Import Nhánh B
from captioning.inference import predict_caption
 
app = FastAPI(title="Medical Lesion Detection API", version="1.0")

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Khởi tạo thư mục
os.makedirs("backend/static", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# --- KHỞI TẠO BỘ NHỚ CACHE ---
analysis_cache = {}

# Load mô hình Nhánh A
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading TorchXRayVision model on {device}...")
vision_model = load_model(device)
print("TorchXRayVision Model loaded successfully!")

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None)
):
    # 1. Đọc nội dung ảnh và tạo mã Hash
    image_bytes = await file.read()
    image_hash = hashlib.md5(image_bytes).hexdigest()
    
    file_location = f"sample_images/{file.filename}"
    
    # Ghi file từ bytes đã đọc
    with open(file_location, "wb") as f:
        f.write(image_bytes)

    answer = None  # Biến lưu kết quả VQA

    # 2. XỬ LÝ ẢNH MỚI (CHƯA CÓ TRONG CACHE)
    if image_hash not in analysis_cache:
        print(f"Bức ảnh mới (Hash: {image_hash}), bắt đầu phân tích...")
        
        # --- Chạy Nhánh A: Nhận diện bệnh ---
        detection_result = detect(
            image_path=file_location,
            model=vision_model,
            device=device,
            threshold=0.55
        )
        has_lesion = detection_result["has_lesion"]
        detected_pathologies = detection_result["pathologies"] if has_lesion else "Không phát hiện tổn thương."
        

    # --- SỬA Ở ĐÂY: Xử lý Heatmap hoặc Ảnh Gốc ---
        heatmap_url = f"/static/{file.filename}" # MẶC ĐỊNH LÀ ẢNH GỐC
        
        # Nhớ copy ảnh gốc vào thư mục static để Frontend có thể truy cập được
        shutil.copyfile(file_location, f"backend/static/{file.filename}")

        # --- Chạy Heatmap (Nếu có bệnh) ---
        if has_lesion:
            heatmap_filename = f"heatmap_bbox_{file.filename}"
            heatmap_path = f"backend/static/{heatmap_filename}"
            try:
                _, bbox_img, _ = generate_focused_heatmap(
                    image_path=file_location,
                    model=vision_model,
                    device=device,
                    target_class_index=detection_result["top_class_index"],
                    layer_name="denseblock3"
                )
                cv2.imwrite(heatmap_path, cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
                heatmap_url = f"/static/{heatmap_filename}"
            except Exception as e:
                print(f"Error generating heatmap: {e}")

        # --- Xử lý Caption và VQA ---
        if not has_lesion:
            # 1. TRƯỜNG HỢP BÌNH THƯỜNG: Trả về câu thông báo cố định theo yêu cầu
            caption = "Bình thường. Không phát hiện dấu hiệu bệnh lý trên ảnh X-quang."
            if question:
                # Chỉ cần chạy VQA nếu người dùng có đặt câu hỏi
                answer = await asyncio.to_thread(answer_question, file_location, question, detected_pathologies)
        else:
            # 2. TRƯỜNG HỢP CÓ BỆNH: Đánh thức model Caption để sinh mô tả chi tiết bệnh lý
            print("Đang gọi model Caption để sinh từ miêu tả...")
            task_caption = asyncio.to_thread(predict_caption, file_location)

            if question:
                # Chạy song song cả Caption (Model LSTM) và VQA (Gemini)
                print("Đang chạy song song Caption và VQA...")
                task_vqa = asyncio.to_thread(answer_question, file_location, question, detected_pathologies)
                caption, answer = await asyncio.gather(task_caption, task_vqa)
            else:
                caption = await task_caption

        # --- LƯU KẾT QUẢ VÀO CACHE (Không lưu câu trả lời VQA) ---
        analysis_cache[image_hash] = {
            "has_lesion": has_lesion,
            "detection": detection_result["pathologies"] if has_lesion else None,
            "severity": detection_result["severity"],
            "heatmap_url": heatmap_url,
            "caption": caption,
            "detected_pathologies": detected_pathologies
        }
        print("Đã phân tích xong và lưu vào Cache!")

    else:
        # 3. NẾU ẢNH ĐÃ CÓ TRONG CACHE (HỎI LẠI CÂU THỨ 2)
        print(f"Sử dụng lại kết quả từ Cache cho ảnh (Hash: {image_hash}).")
        cached = analysis_cache[image_hash]
        
        # Chỉ việc gọi lại Gemini nếu có câu hỏi mới
        if question:
            answer = await asyncio.to_thread(
                answer_question, 
                file_location, 
                question, 
                cached["detected_pathologies"]
            )

    # 4. CHUẨN BỊ RESPONSE CUỐI CÙNG
    cached = analysis_cache[image_hash]
    response_data = {
        "has_lesion": cached["has_lesion"],
        "detection": cached["detection"],
        "severity": cached["severity"],
        "heatmap_url": cached["heatmap_url"],
        "caption": cached["caption"],
        "answer": answer
    }
    
    return JSONResponse(content=response_data, status_code=200)


@app.post("/export-pdf")
async def export_pdf(
    originalImageBase64: str = Form(...),
    heatmapImageBase64: str = Form(...),
    caption: str = Form(...)
):
    """
    Tạo báo cáo PDF từ ảnh và mô tả y khoa.
    """
    try:
        print("[DEBUG] export_pdf called")
        print(f"[DEBUG] Original image base64 length: {len(originalImageBase64)}")
        print(f"[DEBUG] Heatmap image base64 length: {len(heatmapImageBase64)}")
        print(f"[DEBUG] Caption length: {len(caption)}")
        
        # Tạo file PDF
        pdf_buffer = BytesIO()
        page_width, page_height = A4
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        
        # Đăng ký font hỗ trợ Unicode cho Tiếng Việt
        pil_font = None
        fonts_to_try = [
            r'C:\Windows\Fonts\arial.ttf',
            r'C:\Windows\Fonts\calibrib.ttf',
            r'C:\Windows\Fonts\times.ttf',
            r'C:\Windows\Fonts\DejaVuSans.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        ]
        # Register Arial font vào reportlab Canvas
        try:
            arial_font_path = r'C:\Windows\Fonts\arial.ttf'
            if os.path.exists(arial_font_path):
                pdfmetrics.registerFont(TTFont('ArialVN', arial_font_path))
                canvas_font = 'ArialVN'
                print(f"[DEBUG] Registered Arial font for Canvas: {arial_font_path}")
            else:
                canvas_font = 'Courier'
                print("[DEBUG] Arial font not found, using Courier for Canvas")
        except Exception as e:
            canvas_font = 'Courier'
            print(f"[DEBUG] Failed to register Arial font: {e}, using Courier")
        
        # PIL font cho text image
        pil_font = None
        fonts_to_try = [
            r'C:\Windows\Fonts\arial.ttf',
            r'C:\Windows\Fonts\calibrib.ttf',
            r'C:\Windows\Fonts\times.ttf',
            r'C:\Windows\Fonts\DejaVuSans.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        ]
        
        for font_path in fonts_to_try:
            if os.path.exists(font_path):
                try:
                    pil_font = ImageFont.truetype(font_path, 17)
                    print(f"[DEBUG] Successfully loaded PIL font: {font_path}")
                    break
                except Exception as e:
                    print(f"[DEBUG] Failed to load {font_path}: {e}")
        
        if pil_font is None:
            try:
                pil_font = ImageFont.load_default()
                print("[DEBUG] Using default PIL font (bitmap, may not support Vietnamese)")
            except Exception as e:
                print(f"[DEBUG] Cannot load default font: {e}")
                pil_font = None
        
        def safe_draw_string(canvas_obj, x, y, text, font_name=None, font_size=10):
            """Vẽ text trên Canvas (fallback khi PIL text image không thành công)"""
            if font_name is None:
                font_name = canvas_font  # Use registered Arial or fallback Courier
            try:
                canvas_obj.setFont(font_name, font_size)
                canvas_obj.drawString(x, y, text)
            except Exception as e:
                print(f"[WARNING] Font error with {font_name}: {e}")
                try:
                    canvas_obj.setFont("Courier", font_size)
                    canvas_obj.drawString(x, y, text)
                except Exception as e2:
                    print(f"[WARNING] Failed to draw: {e2}")
        
        def draw_vietnamese_text(text, max_width=700):
            """Vẽ text Tiếng Việt bằng PIL, return ảnh"""
            if pil_font is None:
                print("[DEBUG] PIL font is None, cannot draw Vietnamese text")
                return None
            
            try:
                line_height = 24
                words = text.split()
                lines = []
                current_line = ""
                
                # Word wrapping
                for word in words:
                    test_line = (current_line + " " + word).strip()
                    try:
                        bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), test_line, font=pil_font)
                        text_width = bbox[2] - bbox[0]
                    except:
                        text_width = len(test_line) * 8  # Fallback estimate
                    
                    if text_width < max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                lines = lines[:25]  # Max 25 lines
                img_height = max(len(lines) * line_height + 15, 50)
                img = Image.new('RGB', (max_width, img_height), color='white')
                draw = ImageDraw.Draw(img)
                
                y = 8
                for line in lines:
                    try:
                        draw.text((8, y), line, fill=(0, 0, 0), font=pil_font)
                    except Exception as e:
                        print(f"[WARNING] Failed to draw line '{line}': {e}")
                    y += line_height
                
                print(f"[DEBUG] Vietnamese text image created: {max_width}x{img_height}, {len(lines)} lines")
                return img
            except Exception as e:
                print(f"[ERROR] Error in draw_vietnamese_text: {e}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                return None
        
        # Layout constants
        margin_x = 16 * mm
        content_w = page_width - (2 * margin_x)

        # Header
        header_y = page_height - 20 * mm
        safe_draw_string(c, margin_x, header_y, "ChestVision AI - Báo cáo phân tích", font_size=16)
        safe_draw_string(c, margin_x, header_y - 8 * mm, "Hệ thống phát hiện tổn thương trên ảnh X-quang ngực", font_size=10)
        c.setStrokeColorRGB(0.82, 0.85, 0.89)
        c.line(margin_x, header_y - 11 * mm, page_width - margin_x, header_y - 11 * mm)

        # Image section title
        image_title_y = header_y - 18 * mm
        safe_draw_string(c, margin_x, image_title_y, "Hình ảnh phân tích", font_size=11)
        
        # Giải mã base64 và tạo ảnh
        try:
            # Original image
            original_b64 = originalImageBase64
            if original_b64.startswith('data:'):
                original_b64 = original_b64.split(',')[1]
            print(f"[DEBUG] Decoding original image, length: {len(original_b64)}")
            original_img_data = base64.b64decode(original_b64)
            original_img = Image.open(BytesIO(original_img_data))
            print(f"[DEBUG] Original image size: {original_img.size}")
            
            # Heatmap image
            heatmap_b64 = heatmapImageBase64
            # Nếu là URL (từ server), tải từ file thay vì decode base64
            if heatmap_b64.startswith('http'):
                print(f"[DEBUG] Loading heatmap from URL: {heatmap_b64}")
                import requests
                response = requests.get(heatmap_b64)
                if response.status_code == 200:
                    heatmap_img = Image.open(BytesIO(response.content))
                else:
                    raise Exception(f"Failed to load heatmap from URL: {response.status_code}")
            else:
                if heatmap_b64.startswith('data:'):
                    heatmap_b64 = heatmap_b64.split(',')[1]
                print(f"[DEBUG] Decoding heatmap image, length: {len(heatmap_b64)}")
                heatmap_img_data = base64.b64decode(heatmap_b64)
                heatmap_img = Image.open(BytesIO(heatmap_img_data))
            print(f"[DEBUG] Heatmap image size: {heatmap_img.size}")
            
            # Tạo ImageReader từ ảnh gốc (không resize)
            original_buffer = BytesIO()
            original_img.save(original_buffer, format='PNG')
            original_buffer.seek(0)
            original_reader = ImageReader(original_buffer)
            
            # Tạo ImageReader từ heatmap (không resize)
            heatmap_buffer = BytesIO()
            heatmap_img.save(heatmap_buffer, format='PNG')
            heatmap_buffer.seek(0)
            heatmap_reader = ImageReader(heatmap_buffer)
            
            # Two-column image cards
            gap = 8 * mm
            card_w = (content_w - gap) / 2
            card_h = 88 * mm
            card_top = image_title_y - 4 * mm
            left_x = margin_x
            right_x = margin_x + card_w + gap

            # Draw card backgrounds
            c.setStrokeColorRGB(0.85, 0.88, 0.92)
            c.setFillColorRGB(0.98, 0.99, 1.00)
            c.roundRect(left_x, card_top - card_h, card_w, card_h, 3 * mm, stroke=1, fill=1)
            c.roundRect(right_x, card_top - card_h, card_w, card_h, 3 * mm, stroke=1, fill=1)

            # Card labels
            safe_draw_string(c, left_x + 4 * mm, card_top - 6 * mm, "Ảnh X-quang gốc", font_size=10)
            safe_draw_string(c, right_x + 4 * mm, card_top - 6 * mm, "Phát hiện tổn thương (GradCAM)", font_size=10)

            # Image viewport inside cards
            viewport_pad_x = 4 * mm
            viewport_pad_bottom = 4 * mm
            viewport_top_offset = 11 * mm
            viewport_w = card_w - (2 * viewport_pad_x)
            viewport_h = card_h - viewport_top_offset - viewport_pad_bottom

            c.drawImage(
                original_reader,
                left_x + viewport_pad_x,
                (card_top - card_h) + viewport_pad_bottom,
                width=viewport_w,
                height=viewport_h,
                preserveAspectRatio=True,
                anchor='c',
                mask='auto'
            )

            c.drawImage(
                heatmap_reader,
                right_x + viewport_pad_x,
                (card_top - card_h) + viewport_pad_bottom,
                width=viewport_w,
                height=viewport_h,
                preserveAspectRatio=True,
                anchor='c',
                mask='auto'
            )

            desc_title_y = (card_top - card_h) - 8 * mm
            
        except Exception as e:
            print(f"[ERROR] Lỗi xử lý ảnh: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise
        
        # Tiêu đề mô tả
        safe_draw_string(c, margin_x, desc_title_y, "Mô tả y khoa", font_size=12)
        
        # Vẽ mô tả thành ảnh bằng PIL (hỗ trợ Unicode)
        try:
            print(f"[DEBUG] Creating Vietnamese text image for caption: {caption[:50]}...")
            caption_img = draw_vietnamese_text(caption, max_width=1000)
            
            if caption_img is not None:
                # Description container
                desc_box_top = desc_title_y - 4 * mm
                desc_box_h = 58 * mm
                c.setStrokeColorRGB(0.85, 0.88, 0.92)
                c.setFillColorRGB(1.0, 1.0, 1.0)
                c.roundRect(margin_x, desc_box_top - desc_box_h, content_w, desc_box_h, 3 * mm, stroke=1, fill=1)

                # Resize caption image to fit description box
                caption_img.thumbnail((1300, 520), Image.Resampling.LANCZOS)
                caption_buffer = BytesIO()
                caption_img.save(caption_buffer, format='PNG')
                caption_buffer.seek(0)
                caption_reader = ImageReader(caption_buffer)
                
                # Draw caption image centered inside description box
                cap_pad = 4 * mm
                cap_w = content_w - (2 * cap_pad)
                cap_h = desc_box_h - (2 * cap_pad)
                c.drawImage(
                    caption_reader,
                    margin_x + cap_pad,
                    (desc_box_top - desc_box_h) + cap_pad,
                    width=cap_w,
                    height=cap_h,
                    preserveAspectRatio=True,
                    anchor='w',
                    mask='auto'
                )
                print("[DEBUG] Caption image drawn successfully")
            else:
                # Fallback: vẽ text thường nếu PIL không có
                print("[DEBUG] PIL font not available, falling back to text drawing")
                text_y = desc_title_y - 8 * mm
                chars_per_line = 95
                caption_clean = caption.replace('\n', ' ')
                lines = []
                for i in range(0, len(caption_clean), chars_per_line):
                    lines.append(caption_clean[i:i+chars_per_line])
                
                for line in lines[:8]:
                    safe_draw_string(c, margin_x, text_y, line, font_size=11)
                    text_y -= 5 * mm
                    if text_y < 18 * mm:
                        break
        except Exception as e:
            print(f"[ERROR] Failed to create caption image: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            # Fallback: vẽ text thường
            text_y = desc_title_y - 8 * mm
            chars_per_line = 95
            caption_clean = caption.replace('\n', ' ')
            lines = []
            for i in range(0, len(caption_clean), chars_per_line):
                lines.append(caption_clean[i:i+chars_per_line])
            
            for line in lines[:8]:
                safe_draw_string(c, margin_x, text_y, line, font_size=11)
                text_y -= 5 * mm
                if text_y < 18 * mm:
                    break
        
        # Footer
        safe_draw_string(c, margin_x, 12*mm, "Báo cáo được tạo bởi ChestVision AI", font_size=7)
        
        # Save PDF
        c.save()
        pdf_buffer.seek(0)
        
        print("[DEBUG] PDF created successfully")
        
        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=ChestVision_Report.pdf"}
        )
        
    except Exception as e:
        print(f"Lỗi tạo PDF: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return JSONResponse(
            {"error": f"Lỗi tạo PDF: {str(e)}\n{traceback.format_exc()}"},
            status_code=500
        )


# Phục vụ trực tiếp frontend prototype từ gốc project.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")