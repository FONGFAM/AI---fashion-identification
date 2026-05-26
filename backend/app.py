import os
import io
import base64
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# Tùy chọn: tắt log khó chịu của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
# Cho phép tất cả các domain có thể truy cập (CORS)
CORS(app)

# Giới hạn kích thước file upload: 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Tên các loại trang phục theo chuẩn Fashion MNIST (đã Việt hóa dễ hiểu)
CLASS_NAMES = [
    "Áo thun / Áo phông", "Quần dài", "Áo len", 
    "Đầm / Váy liền", "Áo khoác", "Sandal / Dép quai hậu", "Áo sơ mi", 
    "Giày thể thao", "Túi xách", "Bốt cổ thấp"
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fashion_model.keras")
model = None

# Cố gắng load model khi khởi động server
def load_fashion_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Da tai model thanh cong tu {MODEL_PATH} bang load_model()")
            print(f"Model Input Shape: {model.input_shape}")
        except Exception as e:
            print(f"Loi khi tai model: {e}")
            model = None
    else:
        print(f"CANH BAO: Khong tim thay file model tai {MODEL_PATH}. Ung dung se chay o che do gia lap (Mock Mode).")

load_fashion_model()

def preprocess_image(image_bytes):
    """
    Tien xu ly anh nguoi dung tai len -> format Fashion MNIST (28x28 grayscale).
    Fashion MNIST: nen DEN (pixel=0), vat the TRANG (pixel=255).
    Pipeline on dinh, ket qua nhu nhau moi lan voi cung 1 anh.
    """
    image = Image.open(io.BytesIO(image_bytes))

    # 1. Sua EXIF rotation (anh chup tu dien thoai)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # 2. Chuan hoa color mode ve grayscale
    if image.mode in ("CMYK", "P", "LA"):
        image = image.convert("RGB")
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    image = image.convert("L")

    # 3. Auto-crop: loai bo vien trang/den, giu vung san pham chinh
    #    Dung nguong co dinh (khong phu thuoc mean) -> on dinh hon
    arr = np.array(image, dtype=np.float32)
    h, w = arr.shape
    
    # Lấy độ sáng viền ảnh để xác định màu nền chính xác (tránh bị ảnh hưởng bởi kích thước vật thể)
    border_pixels = []
    border_pixels.extend(arr[0:5, :].flatten())
    border_pixels.extend(arr[h-5:h, :].flatten())
    border_pixels.extend(arr[:, 0:5].flatten())
    border_pixels.extend(arr[:, w-5:w].flatten())
    mean_border_brightness = float(np.mean(border_pixels))

    if mean_border_brightness > 128:
        # Nền sáng: vật thể là các pixel tối màu < 240
        content_mask = arr < 240
    else:
        # Nền tối: vật thể là các pixel sáng màu > 15
        content_mask = arr > 15

    rows_ok = np.any(content_mask, axis=1)
    cols_ok = np.any(content_mask, axis=0)

    if rows_ok.any() and cols_ok.any():
        row_idx = np.where(rows_ok)[0]
        col_idx = np.where(cols_ok)[0]
        h, w = arr.shape
        rmin = max(0,   int(row_idx[0])  - 10)
        rmax = min(h-1, int(row_idx[-1]) + 10)
        cmin = max(0,   int(col_idx[0])  - 10)
        cmax = min(w-1, int(col_idx[-1]) + 10)
        # Chi crop neu vung noi dung chiem > 20% dien tich (tranh crop sai)
        crop_ratio = (rmax - rmin) * (cmax - cmin) / (h * w)
        if crop_ratio > 0.2:
            image = image.crop((cmin, rmin, cmax + 1, rmax + 1))

    # 4. Quyet dinh INVERT dua tren anh goc (kiem tra do sang vien cua anh 28x28)
    #    -> Vien anh (border) dai dien cho nen chu khong bi anh huong boi kich thuoc vat the
    arr_raw = np.array(image.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.float32)
    h_raw, w_raw = arr_raw.shape
    border_pixels_raw = []
    border_pixels_raw.extend(arr_raw[0:2, :].flatten())           # Top border
    border_pixels_raw.extend(arr_raw[h_raw-2:h_raw, :].flatten()) # Bottom border
    border_pixels_raw.extend(arr_raw[:, 0:2].flatten())           # Left border
    border_pixels_raw.extend(arr_raw[:, w_raw-2:w_raw].flatten())         # Right border
    
    mean_border_raw = float(np.mean(border_pixels_raw)) / 255.0

    # Neu vien anh sang (mean_border_raw > 0.5) -> nen sang -> can invert de ve nen toi
    should_invert = mean_border_raw > 0.5
    if should_invert:
        image = ImageOps.invert(image)

    # 5. Tang tuong phan SAU khi invert de lam ro silhouette
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # 6. Resize & chuan hoa [0, 1] theo yeu cau cua model
    if model is not None:
        input_shape = model.input_shape  # e.g., (None, 96, 96, 3) or (None, 28, 28, 1)
        h_in = input_shape[1]
        w_in = input_shape[2]
        c_in = input_shape[3] if len(input_shape) == 4 else 1
    else:
        h_in, w_in, c_in = 28, 28, 1

    # Luon luu anh 28x28 de lam anh xem truoc AI o frontend
    image_preview = image.resize((28, 28), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    image_preview.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Resize cho phu hop voi input model
    image = image.resize((w_in, h_in), Image.Resampling.LANCZOS)

    if model is not None and model.name == "fashion_mobilenetv2":
        # MobileNetV2 pre-trained model: convert sang RGB va dung preprocess_input cua MobileNetV2
        image_rgb = image.convert("RGB")
        img_array = np.array(image_rgb, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0) # (1, 96, 96, 3)
    else:
        # Mo hinh CNN thong thuong: grayscale [0, 1]
        img_array = np.array(image) / 255.0
        if c_in == 1:
            img_array = img_array.reshape(1, h_in, w_in, 1)
        else:
            img_array = img_array.reshape(1, h_in, w_in)

    return img_array, img_base64

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không tìm thấy file ảnh!"}), 400
        
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"success": False, "error": "Chưa chọn file!"}), 400
        
    try:
        print(f"\n--- Nhận yêu cầu dự đoán cho file: {file.filename} ---")
        
        # Đọc dữ liệu ảnh
        img_bytes = file.read()
        
        # Tiền xử lý ảnh
        processed_image, img_base64 = preprocess_image(img_bytes)
        
        # Dự đoán
        if model is not None:
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class]) * 100
            
            result_name = CLASS_NAMES[predicted_class]
            print(f"> Kết quả AI: {result_name} ({confidence:.2f}%)")
        else:
            # Chế độ giả lập khi chưa có model
            print("> CẢNH BÁO: Đang chạy chế độ MÔ PHỎNG (Mock mode) do chưa load được model!")
            import random
            result_name = random.choice(CLASS_NAMES)
            confidence = random.uniform(70.0, 99.9)
            print(f"> Kết quả Mock: {result_name} ({confidence:.2f}%)")
            
        return jsonify({
            "success": True,
            "result": result_name,
            "confidence": round(confidence, 2),
            "is_mock": model is None,
            "processed_image": f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        print(f"\n[ERROR] Lỗi trong quá trình xử lý ảnh: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Lỗi xử lý ảnh: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File ảnh quá lớn! Giới hạn tối đa là 16 MB."}), 413

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
