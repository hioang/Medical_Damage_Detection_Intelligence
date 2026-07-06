# vqa/vqa_service.py 
import PIL.Image
import time
from google import genai

# HARDCODE API KEY CHO BÀI TẬP NỘI BỘ
GEMINI_API_KEY = "Them API Key của bạn vào đây"
client = genai.Client(api_key=GEMINI_API_KEY)

def answer_question(image_path: str, question: str, detection_context, max_retries: int = 2) -> str:
    """
    Hàm VQA: Nhận ảnh, câu hỏi và BỐI CẢNH (từ Module 2) để gọi Gemini.
    Không load lại model nhận diện ở đây để tiết kiệm VRAM.
    detection_context có thể là: string, list of dicts, hoặc None
    
    Với retry logic cho lỗi 503 (service overload).
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            img = PIL.Image.open(image_path)
            
            # Xây dựng prompt kết hợp kết quả từ Module 2
            context_str = "Không phát hiện bệnh lý bất thường."
            if detection_context:
                # Nếu detection_context là string, dùng trực tiếp
                if isinstance(detection_context, str):
                    context_str = detection_context if detection_context != "Không phát hiện tổn thương." else "Không phát hiện bệnh lý bất thường."
                # Nếu là list, trích xuất tên bệnh
                elif isinstance(detection_context, list):
                    diseases = [d["name"] for d in detection_context]
                    context_str = f"Các dấu hiệu phát hiện được trên ảnh: {', '.join(diseases)}."

            prompt = f"""
            Bạn là một trợ lý AI phân tích hình ảnh y khoa (X-quang).
            Thông tin chẩn đoán từ hệ thống thị giác máy tính: {context_str}
            
            Dựa vào hình ảnh X-quang được cung cấp và thông tin chẩn đoán trên, hãy trả lời câu hỏi sau một cách ngắn gọn, chính xác bằng tiếng Việt:
            Câu hỏi: {question}
            """
            
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=[prompt, img]
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            # Check for 503 Service Unavailable
            if '503' in error_str or 'UNAVAILABLE' in error_str or 'high demand' in error_str:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff: 2s, 4s, ...
                    print(f"[VQA] API quá tải (503). Thử lại sau {wait_time}s... (lần {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Dịch vụ VQA hiện không khả dụng (Gemini API quá tải). Vui lòng thử lại sau vài phút."
            # Other errors
            if 'INVALID_ARGUMENT' in error_str or 'invalid' in error_str.lower():
                return "Lỗi VQA: Câu hỏi hoặc ảnh không hợp lệ. Vui lòng kiểm tra lại."
            if 'UNAUTHENTICATED' in error_str or 'authentication' in error_str.lower():
                return "Lỗi VQA: Vấn đề xác thực API. Vui lòng liên hệ quản trị viên."
            # Generic error message
            return f"Lỗi VQA: {error_str[:100]}... Vui lòng thử lại sau."
    
    # This shouldn't be reached but just in case
    return "Lỗi VQA: Không thể kết nối sau nhiều lần thử. Vui lòng thử lại sau."
