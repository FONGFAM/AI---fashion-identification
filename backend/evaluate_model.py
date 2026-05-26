import os
import sys

# Fix encoding Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# pyrefly: ignore [missing-import]
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fashion_model.keras")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("=" * 60)
print("  FASHION MNIST MODEL EVALUATION")
print("=" * 60)

# ── Bước 1: Tải model ───────────────────────────────────────────
print(f"\n[1/2] Tải model từ {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("     Model đã được tải thành công!")
    model.summary()
except Exception as e:
    print(f"     Lỗi khi tải model: {e}")
    sys.exit(1)

# ── Bước 4: Đánh giá trên Fashion MNIST test set ────────────────
print("\n[4/4] Tải Fashion MNIST test set và đánh giá...")
(_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Model dùng CNN → cần input shape (28, 28, 1)
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., np.newaxis]  # (10000, 28, 28, 1)

loss, acc = model.evaluate(x_test, y_test, verbose=1)

preds = model.predict(x_test, verbose=0)
pred_classes = np.argmax(preds, axis=1)

print("\n" + "=" * 60)
print("  KẾT QUẢ ĐÁNH GIÁ")
print("=" * 60)
print(f"  Test Loss    : {loss:.4f}")
print(f"  Test Accuracy: {acc * 100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, pred_classes, target_names=CLASS_NAMES))

print("  Confusion Matrix (hàng = thực, cột = dự đoán):")
cm = confusion_matrix(y_test, pred_classes)
header = "         " + "  ".join(f"{n[:5]:>5}" for n in CLASS_NAMES)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:5d}" for v in row)
    print(f"  {CLASS_NAMES[i][:8]:>8}: {row_str}")

print("\n" + "=" * 60)
print("  ĐÁNH GIÁ TỔNG QUAN")
print("=" * 60)
if acc >= 0.92:
    print("  ✅ Tuyệt vời! Model đạt độ chính xác >= 92%")
elif acc >= 0.88:
    print("  🟡 Tốt! Model đạt độ chính xác >= 88%")
elif acc >= 0.80:
    print("  🟠 Khá ổn. Có thể cải thiện thêm.")
else:
    print("  🔴 Độ chính xác còn thấp. Nên train thêm hoặc điều chỉnh model.")

# Per-class accuracy
print("\n  Độ chính xác theo từng class:")
for i, name in enumerate(CLASS_NAMES):
    class_mask = (y_test == i)
    class_acc = (pred_classes[class_mask] == y_test[class_mask]).mean()
    bar = "█" * int(class_acc * 20) + "░" * (20 - int(class_acc * 20))
    print(f"  {name:<20} [{bar}] {class_acc*100:.1f}%")
print("=" * 60)
