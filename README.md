# Phát hiện và Giải mã QR Code dạng OBB (Oriented Bounding Box)

Dự án này cung cấp một pipeline hoàn chỉnh để phát hiện các mã QR bị xoay/nghiêng trong môi trường thực tế sử dụng mạng **YOLOv8-OBB**, sau đó nắn chỉnh hình học và giải mã nội dung bằng cơ chế Cascade kết hợp nhiều bộ lọc ảnh (CLAHE, Denoise, Unsharp Mask) và đa backend (OpenCV, PyZbar, ZXing).

## 1. Cài đặt Môi trường (Environment Setup)

Dự án này sử dụng **Conda** để quản lý môi trường và các thư viện phụ thuộc, đảm bảo tính tái lập (reproducibility) cao nhất. Vui lòng làm theo các bước dưới đây để thiết lập.

### Bước 1.1: Tạo và kích hoạt môi trường Conda
Toàn bộ cấu hình phiên bản Python và các thư viện đã được đóng gói trong file `environment.yml`. Mở Terminal (hoặc Anaconda Prompt) và chạy lệnh:

```bash
# Tạo môi trường mới từ file cấu hình
conda env create -f environment.yml
conda activate ten_moi_truong
```

### Bước 1.2: Cài đặt thư viện hệ thống (Chỉ dành cho Linux/Ubuntu)
Thư viện giải mã `pyzbar` phụ thuộc vào bộ thư viện ZBar của hệ điều hành. Dù Conda đã quản lý rất tốt, nhưng nếu bạn chạy mã trên Ubuntu/Debian hoặc Google Colab và gặp lỗi thiếu thư viện C, hãy chạy thêm:
```bash
sudo apt-get update
sudo apt-get install libzbar0
```

---

## 2. Cấu trúc Thư mục Yêu cầu

Để chạy lệnh mặc định không bị lỗi đường dẫn, hãy đảm bảo thư mục dự án của bạn có cấu trúc cơ bản như sau:
```text
QR_Project/  
│
├── main.py                 # File script chính
├── environment.yml         # File cấu hình môi trường Conda
├── qr/                     # Thư mục chứa dữ liệu
│   ├── images/             # Ảnh gốc (.jpg, .png)
│   ├── public_train.csv    # File danh sách ảnh đầu vào
│   └── output_train.csv    # File Ground-truth chuẩn
│
└── runs/obb/QR_OBB_Training/run_v1/weights/
    └── best.pt             # File trọng số mô hình YOLO-OBB đã huấn luyện
```
*(Lưu ý: Bạn có thể đặt file `best.pt` ở bất kỳ đâu và trỏ đường dẫn cho nó thông qua tham số `--model` khi chạy lệnh).*

---

## 3. Hướng dẫn Chạy chương trình (Usage)

Chương trình được vận hành qua giao diện dòng lệnh (CLI) bằng file `main.py`.

### Kịch bản 1: Chỉ chạy suy luận (Inference) và lưu kết quả
Đầu vào là một file CSV chứa cột `image_id` và `image_path`.
```bash
python main.py --data qr/public_train.csv --output my_predict.csv
```

### Kịch bản 2: Suy luận & Đánh giá ngay với Ground-Truth
Chạy suy luận, lưu kết quả, sau đó tự động đối chiếu IoU với file nhãn gốc để in ra các chỉ số mAP, Precision, Recall, F1.
```bash
python main.py --data qr/public_train.csv --gt qr/output_train.csv
```

### Kịch bản 3: Chỉ Đánh giá (Evaluation Only)
Sử dụng khi bạn đã có sẵn file kết quả dự đoán (ví dụ `output.csv`) và chỉ muốn chạy code tính điểm đánh giá (bỏ qua bước chạy model YOLO).
```bash
python main.py --eval-only --pred output.csv --gt qr/output_train.csv --iou-thr 0.5
```

---

## 4. Các Tham số Dòng lệnh (CLI Arguments)

| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `--data` | `None` | Đường dẫn tới file CSV chứa danh sách ảnh cần dự đoán. |
| `--model` | `./runs/.../best.pt` | Đường dẫn tới trọng số YOLO-OBB. |
| `--output` | `output.csv` | Tên file CSV lưu kết quả đầu ra. |
| `--gt` | `None` | File CSV nhãn gốc (Ground Truth) dùng để tính điểm. |
| `--eval-only` | `False` | Bật cờ này nếu chỉ muốn tính điểm (yêu cầu `--pred` và `--gt`). |
| `--pred` | `None` | File CSV kết quả đã dự đoán (dùng kèm `--eval-only`). |
| `--conf` | `0.25` | Ngưỡng tin cậy (Confidence Threshold) cho YOLO. |
| `--iou-thr` | `0.5` | Ngưỡng IoU để tính toán True Positive / False Positive. |
| `--no-decode` | `False` | Bỏ qua bước giải mã nội dung QR (tiết kiệm thời gian nếu chỉ lấy tọa độ). |
| `--device` | `cpu` | Thiết bị chạy suy luận: `cpu`, `0` (GPU 0), hoặc `cuda:0`. |

---

## 5. Định dạng Dữ liệu (Data Formats)

**1. File CSV Đầu vào (`--data`):**
Bắt buộc phải có 2 cột: `image_id` (Tên định danh) và `image_path` (Đường dẫn tới ảnh).
```csv
image_id,image_path
img_001,qr/images/img_001.jpg
img_002,qr/images/img_002.jpg
```

**2. File CSV Kết quả / Ground Truth:**
Bao gồm tọa độ 4 góc của QR Code theo chiều kim đồng hồ và nội dung giải mã.
```csv
image_id,qr_index,x0,y0,x1,y1,x2,y2,x3,y3,content
img_001,0,120.5,150.0,220.5,150.0,220.5,250.0,120.5,250.0,"[https://example.com](https://example.com)"
```
