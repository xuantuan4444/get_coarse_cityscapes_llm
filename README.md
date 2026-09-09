# Cityscapes Coarse Mask Generation & Training

Repository này cung cấp pipeline để:

1. Chuẩn bị dữ liệu **Cityscapes** và checkpoint **SAM3**.
2. Sử dụng SAM3 kết hợp với cấu hình prompt để tạo **coarse masks**.
3. Lưu kết quả coarse masks thành file `.zip`.
4. Sử dụng coarse masks làm dữ liệu đầu vào để huấn luyện ba mô hình:

   * U-Net
   * U-Net + ASPP
   * U-Net + ASPP + DINOv2

---

## 1. Project Structure

Đảm bảo project có cấu trúc thư mục tương tự:

```text
get_coarse_city_llm/
│
├── data_cityscapes/
│   └── ...                         # Cityscapes dataset
│
├── weight_sam3/
│   └── sam3.pt                     # SAM3 model checkpoint
│
├── adjust_prompt_city_v3.json     # Prompt configuration
│
├── requirements.txt                dependencies
│
├── get_coarse_city_llm.py          # Generate coarse masks
│
├── train_unet_city_llm.py         
├── train_unetaspp_city_llm.py    
└── train_unetasppdinov2_city_llm.py 
```


---

# 2. Prepare Dataset and Model Checkpoint

Trước khi chạy chương trình, cần chuẩn bị **Cityscapes dataset**, **SAM3 checkpoint** và **prompt configuration**.

## 2.1. SAM3 Checkpoint

Tải checkpoint `sam3.pt` và đặt vào:

```text
get_coarse_city_llm/weight_sam3/sam3.pt
```

Download:

https://drive.google.com/file/d/1FiUmJKX-CFvkKdcKecsdwikGJ2kVO7Eu/view?usp=sharing&utm_source=chatgpt.com

Sau khi tải xuống, kiểm tra rằng file tồn tại tại:

```text
weight_sam3/
└── sam3.pt
```

---

## 2.2. Cityscapes Dataset

Tải dataset Cityscapes và đặt toàn bộ dữ liệu vào:

```text
get_coarse_city_llm/data_cityscapes/
```

Download:

https://drive.google.com/file/d/1vEu15A3In1CQ00hFFzgpHCEIR1kWHzjx/view?usp=sharing&utm_source=chatgpt.com

Kiểm tra cấu trúc thư mục sau khi tải dataset.

Nếu dataset được đặt ở vị trí khác với cấu trúc trên, cần cập nhật lại các biến đường dẫn trong:

```text
get_coarse_city_llm.py
```

---

## 2.3. Prompt Configuration

File:

```text
adjust_prompt_city_v3.json
```

được sử dụng làm cấu hình prompt cho pipeline.

Mặc định, file được đặt tại:

```text
get_coarse_city_llm/adjust_prompt_city_v3.json
```

Cần đảm bảo đường dẫn đến file này được cấu hình chính xác trong source code trước khi chạy.

---

# 3. Environment Setup

Khuyến nghị sử dụng **Python virtual environment (`venv`)** hoặc **Conda** để hạn chế xung đột giữa các thư viện.

## 3.1. Create Virtual Environment

### Using `venv`

Tạo environment:

```bash
python -m venv .venv
```

### Windows PowerShell

Kích hoạt environment:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / Ubuntu

```bash
source .venv/bin/activate
```


---

# 4. Install Dependencies

## 4.1. Install SAM3

Cài đặt SAM3 trực tiếp từ repository chính thức:

```bash
pip install "git+https://github.com/facebookresearch/sam3.git"
```

## 4.2. Install Project Dependencies

Sau khi cài đặt SAM3, cài đặt các dependencies còn lại:

```bash
pip install -r requirements.txt
```

Nếu sử dụng GPU, cần đảm bảo môi trường PyTorch và CUDA tương thích với GPU của server.

---

# 5. Generate Coarse Masks

Sau khi hoàn tất:

* Cityscapes dataset
* SAM3 checkpoint
* `adjust_prompt_city_v3.json`
* Python environment
* Required dependencies

có thể bắt đầu tạo coarse masks.

Chạy:

```bash
python get_coarse_city_llm.py
```

Script sẽ sử dụng:

```text
Cityscapes dataset
        │
        ▼
      SAM3
        │
        ▼
Prompt configuration
(adjust_prompt_city_v3.json)
        │
        ▼
Coarse mask generation
        │
        ▼
Output .zip
```

---

# 6. Output Coarse Masks

Sau khi chương trình chạy hoàn tất, các coarse masks sẽ được lưu thành file `.zip`.

Vị trí output được cấu hình bên trong:

```text
get_coarse_city_llm.py
```

Do đó, nếu muốn thay đổi thư mục output, hãy kiểm tra và chỉnh sửa biến cấu hình output trong file này.

Ví dụ:

```text
output/
└── coarse_cache_city_llm.zip
```

**Lưu ý:** Tên file và thư mục output thực tế phụ thuộc vào cấu hình trong `get_coarse_city_llm.py`.

---

# 7. Training

File `.zip` được tạo ở bước trước sẽ được sử dụng làm **input dataset** cho các bước training.

Pipeline training gồm ba mô hình:

```text
Coarse Masks
        │
        ├──────────────► U-Net
        │
        ├──────────────► U-Net + ASPP
        │
        └──────────────► U-Net + ASPP + DINOv2
```

Các script tương ứng:

```text
train_unet_city_llm.py
train_unetaspp_city_llm.py
train_unetasppdinov2_city_llm.py
```

---

## 7.1. Configure Coarse Mask Path

**Trước khi chạy training**, cần mở source code của từng file và điều chỉnh (`path`) đến file coarse masks.

Ví dụ:

```python
INPUT_PATH = "/path/to/coarse_cache_city_llm"
```

Đảm bảo `INPUT_PATH` trỏ chính xác đến file coarse masks được tạo.

Cần kiểm tra đường dẫn trong cả ba file:

```text
train_unet_city_llm.py
train_unetaspp_city_llm.py
train_unetasppdinov2_city_llm.py
```

---

## 7.2. Configure `adjust_prompt_city_v3.json` Path

Ngoài đường dẫn đến file coarse mask, cần kiểm tra đường dẫn đến:

```text
adjust_prompt_city_v3.json
```

Mặc định:

```text
get_coarse_city_llm/
└── adjust_prompt_city_v3.json
```

Trong cả ba file training, cần đảm bảo đường dẫn đến file JSON được cấu hình chính xác.

Ví dụ:

```python
PROMPT_CONFIG_PATH = "adjust_prompt_city_v3.json"
```

Nếu file nằm ở một vị trí khác:

```python
PROMPT_CONFIG_PATH = "/path/to/adjust_prompt_city_v3.json"
```

Cần kiểm tra trong:

```text
train_unet_city_llm.py
train_unetaspp_city_llm.py
train_unetasppdinov2_city_llm.py
```

để đảm bảo cả ba script đều trỏ đúng đến cùng file:

```text
adjust_prompt_city_v3.json
```
---

# 8. Run Training

Sau khi đã cấu hình chính xác:

* Coarse mask `.zip` path
* `adjust_prompt_city_v3.json` path
* Các đường dẫn dataset/model cần thiết

có thể chạy từng mô hình.

## 8.1. Train U-Net

```bash
python train_unet_city_llm.py
```

## 8.2. Train U-Net + ASPP

```bash
python train_unetaspp_city_llm.py
```

## 8.3. Train U-Net + ASPP + DINOv2

```bash
python train_unetasppdinov2_city_llm.py
```

Có thể chạy riêng từng script tùy theo thí nghiệm cần thực hiện.

---

# 9. Recommended Execution Order

Toàn bộ pipeline nên được thực hiện theo thứ tự:

### Step 1 — Prepare dataset

```text
data_cityscapes/
```

### Step 2 — Prepare SAM3 checkpoint

```text
weight_sam3/sam3.pt
```

### Step 3 — Prepare prompt configuration

```text
adjust_prompt_city_v3.json
```

### Step 4 — Install environment

```bash
python -m venv .venv
```

### Step 5 — Install SAM3

```bash
pip install "git+https://github.com/facebookresearch/sam3.git"
```

### Step 6 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 7 — Generate coarse masks

```bash
python get_coarse_city_llm.py
```

### Step 8 — Locate generated `coarse mask`

Kiểm tra output được tạo bởi `get_coarse_city_llm.py`.

### Step 9 — Configure training paths

Cập nhật trong cả ba file:

```text
train_unet_city_llm.py
train_unetaspp_city_llm.py
train_unetasppdinov2_city_llm.py
```

bao gồm:

```text
1. Coarse mask path
2. adjust_prompt_city_v3.json path
```

### Step 10 — Train models

```bash
python train_unet_city_llm.py
python train_unetaspp_city_llm.py
python train_unetasppdinov2_city_llm.py
```

---

cuối cùng sau khi training xong, các mô hình sẽ được lưu lại trong thư mục output tương ứng. lưu file pth