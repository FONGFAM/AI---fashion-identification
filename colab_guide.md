# Hướng dẫn chạy Huấn luyện Model trên Google Colab & Quá trình Phát triển

Tài liệu này giải thích sự khác biệt giữa mô hình hiện tại với mô hình nguyên bản (chỉ train trên Fashion MNIST gốc), đồng thời hướng dẫn cách chạy huấn luyện lại trên Google Colab để đạt kết quả tương đương.

---

## 1. Những cải tiến quan trọng của Mô hình hiện tại

Để giải quyết vấn đề mô hình nguyên bản nhận diện rất kém trên ảnh thực tế (do ảnh tự chụp có ánh sáng phức tạp, viền ảnh không đồng đều, nền sáng/tối khác nhau), chúng ta đã thực hiện các cải tiến sau:

1. **Pipeline Tiền xử lý Ảnh Nâng cao** (trong [app.py](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/app.py)):
   * **Tự động căn chỉnh góc xoay (EXIF rotation)**: Đảm bảo ảnh chụp từ điện thoại không bị ngược/nghiêng.
   * **Tự động cắt vật thể (Auto-crop)**: Sử dụng mặt nạ độ sáng (`content_mask`) để xác định tọa độ biên của sản phẩm thời trang và cắt bỏ phần viền thừa (chiếm tối thiểu 20% diện tích).
   * **Tự động đảo ngược màu nền (Auto-invert)**: Tính toán độ sáng trung bình của viền ảnh 28x28 (`border_pixels_raw`) để quyết định xem ảnh có cần đảo ngược màu hay không. Điều này đảm bảo nền luôn là màu đen (pixel gần 0) và vật thể là màu sáng (pixel gần 255) theo đúng chuẩn Fashion MNIST.
   * **Tăng cường độ tương phản (Contrast Enhancement)**: Làm nổi bật đường nét (silhouette) của trang phục sau khi đảo ảnh.

2. **Tích hợp và Trộn dữ liệu thực tế** (trong [train_model.py](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/train_model.py)):
   * Mô hình đọc toàn bộ các ảnh chụp thực tế trong thư mục `success/` và `fail/`.
   * Sử dụng hàm `guess_label` để tự động gán nhãn dựa trên từ khóa trong tên file hoặc ánh xạ thủ công.
   * **Oversampling (Nhân bản dữ liệu x150 lần)**: Do tập train Fashion MNIST có tới 60,000 ảnh, nếu chỉ thêm vài chục ảnh thực tế thì chúng sẽ bị "chìm" và mô hình không học được. Việc nhân bản giúp tăng trọng số của ảnh thực tế lên khoảng 10% tập train, giúp mô hình tối ưu hóa tốt hơn trên ảnh thực tế.

3. **Cải tiến Kiến trúc CNN**:
   * Thêm các lớp **Batch Normalization** để ổn định quá trình huấn luyện và tránh hiện tượng gradient biến mất.
   * Sử dụng **GlobalAveragePooling2D** và **Dropout** để giảm overfitting trước khi đưa vào các lớp Dense phân loại.

---

## 2. Cách chạy Huấn luyện trên Google Colab

Bạn có thể chạy huấn luyện trên Google Colab theo 2 cách dưới đây.

### Cách 1: Clone từ GitHub (Khuyên dùng và Dễ nhất)
Nếu bạn đã đẩy code dự án lên một kho lưu trữ GitHub (ví dụ: `https://github.com/username/AI-fashion-identification`), hãy tạo một notebook mới trên Google Colab và chạy các lệnh sau:

```python
# 1. Clone mã nguồn từ GitHub
!git clone https://github.com/username/AI-fashion-identification.git
%cd AI-fashion-identification

# 2. Cài đặt các thư viện cần thiết
!pip install tensorflow pillow flask flask-cors numpy scikit-learn

# 3. Chạy file train trực tiếp
!python backend/train_model.py --epochs 30
```

### Cách 2: Tải lên file ZIP dữ liệu `uploads.zip` (Nếu chạy file lẻ)
Nếu bạn muốn viết một file script Colab tự chứa (self-contained) mà không cần clone git:

1. Tải file [uploads.zip](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/uploads.zip) (đã được tạo sẵn trong thư mục `backend/`) về máy tính của bạn.
2. Mở Google Colab, tạo một notebook mới.
3. Ở thanh bên trái, nhấn vào biểu tượng Thư mục và tải file `uploads.zip` lên thư mục gốc của Colab.
4. Tạo một ô mã lệnh (Code cell) và dán đoạn code sau để giải nén và tạo cấu trúc thư mục:

```python
# 1. Giải nén dữ liệu uploads
import zipfile
import os

with zipfile.ZipFile("uploads.zip", "r") as zip_ref:
    zip_ref.extractall("uploads")

# Tạo thư mục model để lưu kết quả
os.makedirs("model", exist_ok=True)
```

5. Định nghĩa lại toàn bộ Pipeline Tiền xử lý, Trộn dữ liệu và Huấn luyện trong một Code cell trên Colab. Bạn có thể sao chép trực tiếp nội dung từ file [train_model.py](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/train_model.py) kết hợp với các hàm phụ trợ từ [app.py](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/app.py) và [evaluate_uploads.py](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/evaluate_uploads.py) để chạy trực tiếp trên Colab.

6. Sau khi chạy xong, tải file `fashion_model.keras` được lưu ở thư mục `model/` trên Colab về và ghi đè vào thư mục [model/](file:///e:/Applications/Workspace/Project/AI---fashion-identification/backend/model/) ở máy local của bạn.
