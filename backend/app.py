import os
import io
# pyrefly: ignore [missing-import]
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Tùy chọn: tắt log khó chịu của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
# Cho phép tất cả các domain có thể truy cập (CORS)
CORS(app)

# Tên các loại trang phục theo chuẩn Fashion MNIST
CLASS_NAMES = [
    "Áo thun (T-shirt/top)", "Quần dài (Trouser)", "Áo len (Pullover)", 
    "Váy (Dress)", "Áo khoác (Coat)", "Sandal", "Áo sơ mi (Shirt)", 
    "Giày thể thao (Sneaker)", "Túi xách (Bag)", "Bốt (Ankle boot)"
]

MODEL_PATH = os.path.join("model", "fashion_model.h5")
model = None

# Cố gắng load model khi khởi động server
def load_fashion_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Da tai model thanh cong tu {MODEL_PATH}")
        except Exception as e:
            print(f"Loi khi tai model: {e}")
    else:
        print(f"CANH BAO: Khong tim thay file model tai {MODEL_PATH}. Ung dung se chay o che do gia lap (Mock Mode).")

load_fashion_model()

def preprocess_image(image_bytes):
    """
    Tiền xử lý ảnh người dùng tải lên để giống với format Fashion MNIST:
    - Chuyển sang Grayscale (trắng đen)
    - Thay đổi kích thước về 28x28
    - Chuẩn hóa giá trị pixel về [0, 1]
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    # Chuyển sang hệ màu xám
    image = image.convert("L")
    
    # Resize về 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Chuyển thành numpy array và chuẩn hóa
    img_array = np.array(image) / 255.0
    
    # Reshape để phù hợp với input của Keras model
    img_array = img_array.reshape(1, 28, 28)
    
    return img_array

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không tìm thấy file ảnh!"}), 400
        
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"success": False, "error": "Chưa chọn file!"}), 400
        
    try:
        # Đọc dữ liệu ảnh
        img_bytes = file.read()
        
        # Tiền xử lý ảnh
        processed_image = preprocess_image(img_bytes)
        
        # Dự đoán
        if model is not None:
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class]) * 100
            
            result_name = CLASS_NAMES[predicted_class]
        else:
            # Chế độ giả lập khi chưa có model
            import random
            result_name = random.choice(CLASS_NAMES)
            confidence = random.uniform(70.0, 99.9)
            
        return jsonify({
            "success": True,
            "result": result_name,
            "confidence": round(confidence, 2),
            "is_mock": model is None
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
