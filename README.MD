# DỰ ÁN NHẬN DIỆN THỜI TRANG THÔNG MINH (AI FASHION IDENTIFICATION)

Dự án phát triển hệ thống nhận diện sản phẩm thời trang (áo thun, quần dài, váy đầm, giày dép, túi xách...) tự động bằng trí tuệ nhân tạo. Hệ thống kết hợp mạng nơ-ron tích chập (CNN) được huấn luyện nâng cao trên bộ dữ liệu Fashion MNIST cùng với một pipeline tiền xử lý ảnh thực tế thông minh để đạt hiệu suất cao trên các ảnh chụp thực tế từ điện thoại.

---

## 1. Các Tính Năng Nổi Bật (Highlights)

Đối với các mô hình nhận diện ảnh chuẩn như Fashion MNIST, việc nhận diện ảnh chụp thực tế thường rất kém do sự phức tạp về ánh sáng, màu nền và góc chụp. Dự án này giải quyết triệt để vấn đề đó bằng **Pipeline Tiền xử lý Ảnh Nâng cao (Advanced Preprocessing Pipeline)** trước khi đưa vào mô hình AI:

*   **Tự động căn chỉnh hướng ảnh (EXIF Rotation)**: Tự phát hiện và sửa góc xoay của ảnh chụp từ các thiết bị di động để ảnh luôn hướng thẳng đứng.
*   **Tự động cắt vật thể (Auto-crop)**: Tự tính toán phân vùng biên của quần áo/giày dép để cắt bỏ tối đa các viền thừa không liên quan (yêu cầu vật thể chiếm tối thiểu 20% khung hình).
*   **Tự động đảo ngược màu nền (Auto-invert)**: Đảm bảo độ tương thích tuyệt đối với tập dữ liệu Fashion MNIST (nền đen pixel gần 0, vật thể màu sáng pixel gần 255). Hệ thống tự động phát hiện nếu nền ảnh là màu sáng (>128/255) để đảo màu nền sang tối.
*   **Tăng cường độ tương phản (Contrast Enhancement)**: Làm nổi bật silhouette (đường nét phác họa) của trang phục sau khi đảo ảnh để AI nhận biết đặc trưng tốt hơn.
*   **Trộn và tăng cường dữ liệu thực tế (Data Augmentation & Oversampling)**: Trộn các ảnh chụp thực tế vào bộ dữ liệu train gốc và nhân bản (oversampling) gấp 150 lần giúp mô hình không bị "chìm" đặc trưng giữa 60,000 ảnh gốc.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
AI---fashion-identification/
├── backend/
│   ├── app.py                      # Flask API Server (xử lý tiền xử lý và dự đoán trực tuyến)
│   ├── train_model.py              # Script huấn luyện mô hình CNN (hỗ trợ Transfer Learning)
│   ├── evaluate_model.py           # Đánh giá mô hình trên 10,000 ảnh test của Fashion MNIST
│   ├── evaluate_uploads.py         # Kiểm thử và đo lường độ chính xác trên thư mục ảnh thực tế uploads/
│   ├── plot_evaluation.py          # Xuất báo cáo đồ thị trực quan (Confusion Matrix, Loss/Acc curves)
│   ├── model/
│   │   ├── fashion_model.keras    # File lưu trữ trọng số mô hình AI đã huấn luyện
│   │   ├── history.json            # Lưu lịch sử huấn luyện (loss, accuracy qua các epochs)
│   │   ├── classification_report.txt # Báo cáo chi tiết các chỉ số Precision, Recall, F1-score
│   │   └── model_evaluation_charts.png # Đồ thị trực quan 2x2 đánh giá mô hình
│   └── uploads/
│       ├── success/                # Thư mục ảnh thực tế nhận diện đúng
│       └── fail/                   # Thư mục ảnh thực tế từng nhận diện sai (dùng để tối ưu)
├── frontend/
│   ├── css/
│   │   └── style.css               # Giao diện Modern Glassmorphism của ứng dụng
│   ├── js/
│   │   └── main.js                 # Xử lý upload ảnh, kéo thả, gọi API và render đồ thị AI
│   └── index.html                  # Giao diện chính của ứng dụng Web
├── requirements.txt                # Danh sách thư viện Python cần cài đặt
├── progress_report.md              # Báo cáo tiến độ dự án bằng tiếng Việt
└── colab_guide.md                  # Hướng dẫn chạy huấn luyện trên môi trường đám mây Google Colab
```

---

## 3. Hướng Dẫn Cài Đặt Chi Tiết

Dự án yêu cầu cài đặt **Python 3.9, 3.10 hoặc 3.11** (Khuyên dùng **Python 3.11** để cài đặt TensorFlow ổn định nhất trên cả Windows, macOS Intel và Linux).

### Bước 1: Clone dự án và di chuyển vào thư mục
```bash
git clone <URL_REPOSITORY>
cd AI---fashion-identification
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
*   **Trên Windows (PowerShell / Command Prompt)**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **Trên macOS / Linux (Terminal)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Hướng Dẫn Khởi Chạy Ứng Dụng

Sau khi cài đặt xong thư viện, bạn có thể khởi chạy ứng dụng chạy thử trên máy cục bộ (local):

### 1. Khởi động Backend (Flask API)
Đảm bảo bạn đã kích hoạt môi trường ảo `venv` và đang đứng tại thư mục gốc của dự án:
```bash
python backend/app.py
```
Server backend sẽ chạy tại địa chỉ `http://127.0.0.1:5000`.

### 2. Khởi động Frontend (Web UI)
*   **Cách 1 (Nhanh nhất)**: Click đúp trực tiếp vào file `frontend/index.html` để mở giao diện trên trình duyệt web.
*   **Cách 2 (Khuyên dùng - Dùng Server tĩnh Python)**:
    Mở một tab Terminal mới, kích hoạt venv và chạy lệnh:
    ```bash
    python -m http.server 8000 --directory frontend
    ```
    Sau đó truy cập trình duyệt tại địa chỉ: `http://localhost:8000`

---

## 5. Công Cụ Huấn Luyện & Đánh Giá Mô Hình (AI Tools)

Dự án đi kèm các bộ công cụ mạnh mẽ để bạn huấn luyện lại hoặc xuất báo cáo phục vụ thuyết trình/nghiên cứu:

### 1. Huấn luyện lại mô hình (Training)
Nếu bạn muốn bổ sung thêm ảnh thực tế vào thư mục `uploads/` và huấn luyện lại:
```bash
python backend/train_model.py --epochs 30
```
*Bạn có thể chuyển đổi sang kiến trúc Transfer Learning (sử dụng MobileNetV2 pre-trained) bằng cách thêm tham số `--mode transfer`.*

### 2. Đánh giá và xuất sơ đồ trực quan (Evaluation & Plotting)
Để tự động đánh giá mô hình trên tập kiểm thử 10,000 ảnh và tự động xuất sơ đồ báo cáo 2x2:
```bash
python backend/plot_evaluation.py
```
Lệnh này sẽ tự động phân tích và ghi đè hai tệp báo cáo vào thư mục `backend/model/`:
*   `model_evaluation_charts.png`: Đồ thị 2x2 gồm:
    1.  **Ma trận nhầm lẫn (Confusion Matrix)**: Xem AI hay đoán nhầm nhãn nào với nhãn nào.
    2.  **Độ chính xác theo từng lớp (Per-class Accuracy)**: Tỷ lệ nhận diện đúng của từng loại sản phẩm.
    3.  **Đồ thị Accuracy (Accuracy Curves)**: Biến thiên độ chính xác qua các epochs train.
    4.  **Đồ thị Loss (Loss Curves)**: Biến thiên hàm mất mát qua các epochs train.
*   `classification_report.txt`: Báo cáo chi tiết các chỉ số Precision (độ chuẩn xác), Recall (độ phủ), F1-score của mô hình.

### 3. Đánh giá nhanh trên ảnh thực tế
Để kiểm tra độ chính xác cụ thể trên các tập ảnh thực tế trong thư mục `uploads/`:
```bash
python backend/evaluate_uploads.py
```
Lệnh này sẽ hiển thị bảng thống kê chi tiết từng ảnh chụp thực tế đã được dự đoán đúng hay sai với độ tin cậy bao nhiêu phần trăm.
