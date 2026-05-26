"""
train_model.py - Train Fashion MNIST model
Cach dung:
    python -X utf8 backend/train_model.py
    python -X utf8 backend/train_model.py --mode transfer
    python -X utf8 backend/train_model.py --epochs 30
"""
import os, sys, argparse
# pyrefly: ignore [missing-import]
import numpy as np

# Fix encoding Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Arguments ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",   choices=["cnn", "transfer"], default="cnn")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch",  type=int, default=128)
args = parser.parse_args()

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fashion_model.keras")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("=" * 60)
print(f"  FASHION MNIST TRAINING  [{args.mode.upper()}]")
print("=" * 60)

# ── 1. Load data ───────────────────────────────────────────────────
print("\n[1/4] Load Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(f"      Train: {len(x_train):,}  |  Test: {len(x_test):,}")

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., np.newaxis]   # (60000, 28, 28, 1)
x_test  = x_test[..., np.newaxis]

# --- Trộn thêm dữ liệu thực tế từ uploads/ ---
try:
    print("      Dang doc va tron du lieu thuc te tu uploads/...")
    import app
    from evaluate_uploads import guess_label
    
    # Backup và tạm thời ẩn model của app để nhận ảnh dạng 28x28x1
    old_app_model = app.model
    app.model = None
    
    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    x_custom = []
    y_custom = []
    
    for sub in ["success", "fail"]:
        sub_dir = os.path.join(uploads_dir, sub)
        if os.path.exists(sub_dir):
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif')):
                    fpath = os.path.join(sub_dir, fname)
                    lbl = guess_label(fname)
                    if lbl is not None:
                        try:
                            with open(fpath, "rb") as f:
                                img_b = f.read()
                            img_arr, _ = app.preprocess_image(img_b)
                            img_arr = img_arr.reshape(28, 28, 1)
                            x_custom.append(img_arr)
                            y_custom.append(lbl)
                        except Exception as e:
                            pass
                            
    # Khôi phục model của app
    app.model = old_app_model
    
    if len(x_custom) > 0:
        x_custom = np.array(x_custom, dtype="float32")
        y_custom = np.array(y_custom, dtype="int32")
        print(f"      Da doc thanh cong {len(x_custom)} anh thuc te.")
        
        # Nhân bản ảnh thực tế lên 150 lần để có trọng số lớn hơn (tránh bị chìm giữa 60,000 ảnh gốc)
        num_repeats = 150
        x_custom_rep = np.repeat(x_custom, num_repeats, axis=0)
        y_custom_rep = np.repeat(y_custom, num_repeats, axis=0)
        print(f"      Over-sampling: nhan ban x{num_repeats} -> {len(x_custom_rep):,} anh thuc te moi.")
        
        x_train = np.concatenate([x_train, x_custom_rep], axis=0)
        y_train = np.concatenate([y_train, y_custom_rep], axis=0)
        print(f"      Sau khi tron: Train = {len(x_train):,}")
    else:
        print("      Khong tim thay anh thuc te hop le de gop.")
except Exception as e:
    print(f"      Loi khi gop du lieu thuc te: {e}")

# Validation split thu cong (10%)
val_size  = 6000
x_val, y_val     = x_train[:val_size], y_train[:val_size]
x_train2, y_train2 = x_train[val_size:], y_train[val_size:]
print(f"      Train thuc: {len(x_train2):,}  |  Val: {len(x_val):,}")

# ── 2. Augmentation bang tf.data (on-the-fly, khong gay NaN) ──────
print("\n[2/4] Cau hinh Augmentation (tf.data)...")

@tf.function
def augment(image, label):
    # Random flip ngang
    image = tf.image.random_flip_left_right(image)
    # Random brightness nhe (tranh NaN voi BatchNorm)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Random contrast nhe
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (tf.data.Dataset.from_tensor_slices((x_train2, y_train2))
            .shuffle(10000, seed=42)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(args.batch)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
          .batch(args.batch)
          .prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
           .batch(args.batch)
           .prefetch(AUTOTUNE))

print("      OK: flip, brightness, contrast")

# ── 3. Build model ─────────────────────────────────────────────────
print(f"\n[3/4] Build model [{args.mode}]...")

if args.mode == "transfer":
    # MobileNetV2: can resize len 96x96 RGB
    print("      Resize 28->96, grayscale->RGB...")

    @tf.function
    def prepare_transfer(image, label):
        image = tf.image.resize(image, [96, 96])
        image = tf.image.grayscale_to_rgb(image)
        image = keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
        return image, label

    train_ds_t = (tf.data.Dataset.from_tensor_slices((x_train2, y_train2))
                  .shuffle(10000, seed=42)
                  .map(lambda x, y: augment(x, y))
                  .map(prepare_transfer, num_parallel_calls=AUTOTUNE)
                  .batch(args.batch).prefetch(AUTOTUNE))

    val_ds_t   = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
                  .map(prepare_transfer, num_parallel_calls=AUTOTUNE)
                  .batch(args.batch).prefetch(AUTOTUNE))

    test_ds_t  = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                  .map(prepare_transfer, num_parallel_calls=AUTOTUNE)
                  .batch(args.batch).prefetch(AUTOTUNE))

    base = keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    print(f"      MobileNetV2 loaded, {len(base.layers)} layers (frozen)")

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ], name="fashion_mobilenetv2")

    train_ds_fit = train_ds_t
    val_ds_fit   = val_ds_t
    eval_ds      = test_ds_t
    LR = 1e-3

else:
    # CNN tu build, khong dung ImageDataGenerator (tranh bug brightness)
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="fashion_cnn_v2")

    train_ds_fit = train_ds
    val_ds_fit   = val_ds
    eval_ds      = test_ds
    LR = 1e-3

model.summary()
print(f"\n      Params: {model.count_params():,}")

# ── 4. Train ───────────────────────────────────────────────────────
print(f"\n[4/4] Train ({args.epochs} epochs, lr={LR})...")

model.compile(
    optimizer = keras.optimizers.Adam(LR),
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=7,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
]

print(f"\n{'─'*60}")
history = model.fit(
    train_ds_fit,
    validation_data = val_ds_fit,
    epochs          = args.epochs,
    callbacks       = callbacks,
    verbose         = 1
)

# Lưu history training
try:
    import json
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    history_path = os.path.join(os.path.dirname(MODEL_PATH), "history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f)
    print(f"      Da luu lich su huan luyen vao {history_path}")
except Exception as e:
    print(f"      Loi khi luu history: {e}")

# Fine-tune cho transfer learning
if args.mode == "transfer":
    print("\n  [Fine-tune] Mo khoa 30 layers cuoi MobileNetV2...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds_fit, validation_data=val_ds_fit,
              epochs=10, callbacks=callbacks, verbose=1)

# ── Ket qua ────────────────────────────────────────────────────────
loss, acc = model.evaluate(eval_ds, verbose=0)
preds       = model.predict(eval_ds, verbose=0)
pred_cls    = np.argmax(preds, axis=1)

print("\n" + "=" * 60)
print("  KET QUA")
print("=" * 60)
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")
print(f"  Model saved   : {MODEL_PATH}")
print()
print(f"  {'Class':<22} {'Acc':>6}  Bar")
print(f"  {'─'*50}")
for i, name in enumerate(CLASS_NAMES):
    mask    = (y_test == i)
    ca      = float((pred_cls[mask] == i).mean())
    bar     = "█" * int(ca*20) + "░" * (20 - int(ca*20))
    flag    = "OK" if ca >= 0.9 else ("--" if ca >= 0.8 else "!!")
    print(f"  {name:<22} {ca*100:>5.1f}%  {bar} {flag}")

print("=" * 60)
if acc >= 0.92:
    print("  Tuyet voi! >= 92%")
elif acc >= 0.88:
    print("  Tot! >= 88%")
else:
    print("  Thu: python -X utf8 backend/train_model.py --mode transfer")

print("\n  Restart Flask de dung model moi:")
print("  .\\venv\\Scripts\\python.exe backend/app.py")
print("=" * 60)
