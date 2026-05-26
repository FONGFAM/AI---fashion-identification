import os
import sys
import base64

# Fix encoding Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

# Thêm thư mục hiện tại vào sys.path để import app.py
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    from app import preprocess_image, CLASS_NAMES, model
except ImportError as e:
    print(f"Lỗi khi import app.py: {e}")
    sys.exit(1)

def guess_label(filename):
    """
    Đoán nhãn sản phẩm thời trang từ tên file (dựa trên tiền tố chuẩn hóa).
    """
    fn = filename.lower()
    prefixes = {
        "tshirt_": 0,
        "trouser_": 1,
        "pullover_": 2,
        "dress_": 3,
        "coat_": 4,
        "sandal_": 5,
        "shirt_": 6,
        "sneaker_": 7,
        "bag_": 8,
        "ankleboot_": 9
    }
    for pref, idx in prefixes.items():
        if fn.startswith(pref):
            return idx
    return None

def evaluate_folder(folder_path, is_success_folder=True):
    print(f"\n--- Đánh giá thư mục: {os.path.basename(folder_path)} ---")
    if not os.path.exists(folder_path):
        print("Thư mục không tồn tại!")
        return 0, 0
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif'))]
    if not files:
        print("Không tìm thấy file ảnh nào.")
        return 0, 0
    
    correct_count = 0
    total_count = 0
    
    print(f"{'STT':<4} {'Tên File':<40} {'Nhãn Thực':<20} {'AI Dự Đoán':<20} {'Độ tin cậy':<10} {'Kết quả':<8}")
    print("-" * 110)
    
    for idx, filename in enumerate(sorted(files), 1):
        file_path = os.path.join(folder_path, filename)
        
        # Đọc dữ liệu ảnh
        with open(file_path, "rb") as f:
            img_bytes = f.read()
            
        try:
            # Chạy tiền xử lý
            processed_image, _ = preprocess_image(img_bytes)
            
            # Dự đoán
            if model is not None:
                predictions = model.predict(processed_image, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class]) * 100
                pred_name = CLASS_NAMES[predicted_class]
            else:
                pred_name = "Mock (Không load được model)"
                predicted_class = -1
                confidence = 0.0
                
            # Xác định nhãn thực tế
            actual_class_idx = guess_label(filename)
            
            # Nếu là thư mục success và không tự động đoán được nhãn, mặc định nhãn thực = nhãn AI (vì người dùng đã phân loại đúng)
            if actual_class_idx is None and is_success_folder:
                actual_class_idx = predicted_class
                
            actual_name = CLASS_NAMES[actual_class_idx] if actual_class_idx is not None else "Không rõ"
            
            # So sánh kết quả
            is_correct = (predicted_class == actual_class_idx) if actual_class_idx is not None else False
            status_str = "ĐÚNG" if is_correct else "SAI"
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Rút ngắn tên file hiển thị
            display_name = filename if len(filename) <= 38 else filename[:35] + "..."
            print(f"{idx:<4} {display_name:<40} {actual_name:<20} {pred_name:<20} {confidence:>8.2f}% {status_str:<8}")
            
        except Exception as e:
            print(f"{idx:<4} {filename[:38]:<40} Lỗi xử lý: {str(e)}")
            
    print("-" * 110)
    acc = (correct_count / total_count * 100) if total_count > 0 else 0
    print(f"Tổng cộng: {correct_count}/{total_count} đúng. Tỷ lệ chính xác: {acc:.2f}%")
    return correct_count, total_count

if __name__ == "__main__":
    if model is None:
        print("CẢNH BÁO: Không tìm thấy model thực tế, đang chạy ở chế độ MOCK.")
        
    uploads_dir = os.path.join(backend_dir, "uploads")
    success_dir = os.path.join(uploads_dir, "success")
    fail_dir = os.path.join(uploads_dir, "fail")
    
    total_correct = 0
    total_files = 0
    
    c_s, t_s = evaluate_folder(success_dir, is_success_folder=True)
    total_correct += c_s
    total_files += t_s
    
    c_f, t_f = evaluate_folder(fail_dir, is_success_folder=False)
    total_correct += c_f
    total_files += t_f
    
    overall_acc = (total_correct / total_files * 100) if total_files > 0 else 0
    print("\n" + "=" * 60)
    print("                    ĐÁNH GIÁ CHUNG")
    print("=" * 60)
    print(f"  Tổng số ảnh đã test:  {total_files} ảnh")
    print(f"  Số ảnh đoán đúng:    {total_correct} ảnh")
    print(f"  Độ chính xác chung:   {overall_acc:.2f}%")
    print("=" * 60)
