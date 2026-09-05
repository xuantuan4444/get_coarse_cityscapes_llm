# Hướng dẫn Cài đặt và Sử dụng

## 1. Chuẩn bị dữ liệu và cấu trúc thư mục

Trước khi chạy chương trình, hãy đảm bảo dataset, model checkpoint và các file cấu hình được đặt đúng vị trí theo cấu trúc thư mục sau:

```text
get_coarse_city_llm/
├── data_cityscapes/              # Cityscapes dataset
├── weight_sam3/
│   └── sam3.pt                   # SAM3 model checkpoint
├── adjust_prompt.json            # Prompt configuration
├── requirements.txt              # Python dependencies
└── get_coarse_city_llm.py        # Main script
```

### 1.1. SAM3 checkpoint

Tải checkpoint `sam3.pt` và đặt vào thư mục:

```text
get_coarse_city_llm/weight_sam3/sam3.pt
```

**Download:**
https://drive.google.com/file/d/1FiUmJKX-CFvkKdcKecsdwikGJ2kVO7Eu/view?usp=sharing

### 1.2. Cityscapes dataset

Tải dataset Cityscapes và đặt toàn bộ dữ liệu vào:

```text
get_coarse_city_llm/data_cityscapes/
```

**Download:**
https://drive.google.com/file/d/1vEu15A3In1CQ00hFFzgpHCEIR1kWHzjx/view?usp=sharing

> **Lưu ý:** Nếu cấu trúc thư mục hoặc đường dẫn dữ liệu khác với cấu trúc trên, hãy kiểm tra và cập nhật các biến đường dẫn trong `get_coarse_city_llm.py` trước khi chạy chương trình.

---

## 2. Cài đặt môi trường và thư viện

Khuyến nghị sử dụng **Python virtual environment (`venv`) hoặc Conda** để tránh xung đột giữa các thư viện.

### 2.1. Tạo virtual environment

Ví dụ với `venv`:

```bash
python -m venv .venv
```

Kích hoạt môi trường:

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / Ubuntu:**

```bash
source .venv/bin/activate
```

### 2.2. Cài đặt SAM3

Cài đặt SAM3 trực tiếp từ repository chính thức trên GitHub:

```bash
pip install "git+https://github.com/facebookresearch/sam3.git"
```

### 2.3. Cài đặt các thư viện phụ thuộc

Sau khi cài đặt SAM3, cài đặt các dependency còn lại:

```bash
pip install -r requirements.txt
```


---

## 3. Chạy chương trình



Chạy chương trình bằng:

```bash
python get_coarse_city_llm.py
```

---

## 4. Kết quả

Sau khi chương trình hoàn thành, các kết quả coarse mask sẽ được lưu vào thư mục output được cấu hình trong:

```text
get_coarse_city_llm.py
```

Kiểm tra biến cấu hình output trong file nếu muốn thay đổi vị trí lưu kết quả.



