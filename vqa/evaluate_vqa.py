# vqa/evaluate_vqa.py 
import json
import time
from vqa_service import answer_question
from google import genai

# Dùng chung API Key
GEMINI_API_KEY = "Them API Key của bạn vào đây"
judge_client = genai.Client(api_key=GEMINI_API_KEY)

def evaluate_with_llm(ground_truth: str, ai_answer: str) -> bool:
    """
    Dùng Gemini làm Giám khảo (LLM-as-a-Judge) để đánh giá độ chính xác.
    Trả về True nếu đúng ngữ nghĩa, False nếu sai.
    """
    prompt = f"""
    Bạn là một bác sĩ chấm điểm bài thi. Nhiệm vụ của bạn là so sánh CÂU TRẢ LỜI CỦA AI với ĐÁP ÁN CHUẨN.
    
    ĐÁP ÁN CHUẨN: "{ground_truth}"
    CÂU TRẢ LỜI CỦA AI: "{ai_answer}"
    
    CÂU TRẢ LỜI CỦA AI có truyền đạt đúng ý nghĩa y khoa của ĐÁP ÁN CHUẨN không? 
    (Bỏ qua lỗi chính tả nhỏ, chỉ quan tâm đến tính chính xác về mặt y khoa).
    
    Bạn CHỈ ĐƯỢC PHÉP trả lời bằng 1 từ duy nhất: ĐÚNG hoặc SAI.
    """
    try:
        response = judge_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        result = response.text.strip().upper()
        return "ĐÚNG" in result
    except Exception as e:
        print(f"Lỗi khi gọi Giám khảo LLM: {e}")
        return False

def evaluate_system():
    # Load dữ liệu mẫu (Đảm bảo file vqa_rad_data.json nằm cùng thư mục)
    with open('vqa_rad_data.json', 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    test_data = gt_data[:21]
    correct_count = 0
    total = len(test_data)

    print(f"--- BẮT ĐẦU ĐỐI CHIẾU {total} MẪU (LLM-as-a-Judge) ---")

    for i, item in enumerate(test_data):
        img_path = item['image_path']
        question = item['question']
        ground_truth = item['label']

        # 1. Lấy câu trả lời từ hệ thống VQA
        # Lưu ý: Trong đánh giá offline, ta truyền detection_context = None 
        # (hoặc bạn có thể gọi hàm detect ở đây nếu muốn test cả luồng)
        ai_answer = answer_question(img_path, question, detection_context=None)

        # 2. Dùng LLM Giám khảo để chấm điểm
        is_correct = evaluate_with_llm(ground_truth, ai_answer)
        
        if is_correct:
            correct_count += 1

        status = '✅ ĐÚNG' if is_correct else '❌ SAI'
        print(f"[{i+1}/{total}] AI: {ai_answer[:60]}... | {status}")
        
        # NGHỈ 5 GIÂY: Để tránh lỗi 429 (Resource Exhausted)
        time.sleep(5)

    accuracy = (correct_count / total) * 100
    print(f"\n--- BÁO CÁO CUỐI CÙNG ---")
    print(f"Tổng số mẫu test: {total}")
    print(f"Số câu trả lời đúng (Ngữ nghĩa): {correct_count}")
    print(f"Tỉ lệ chuẩn (Accuracy): {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_system()