# Hệ thống phân tích video giao thông đa luồng

## Mô tả chức năng
- Phân tích đồng thời nhiều video CCTV trong một thư mục.
- Phát hiện, theo dõi, gán ID, nhận diện hướng di chuyển và phát hiện phương tiện đi ngược chiều.
- Overlay kết quả (ID, hướng, trạng thái, zone, ...) lên từng video.
- Lưu log vi phạm (đi ngược chiều) vào một file JSON duy nhất.

## Cấu trúc thư mục
```
ORBRO/
├── detection.py           # File chính để chạy phân tích
├── utils.py               # Các module hỗ trợ (YOLO, tracking, direction...)
├── polygons.json          # Định nghĩa các vùng (zone) và hướng hợp lệ cho từng video
├── yolo11s.pt             # File model YOLOv8 (hoặc yolov8s.pt...)
├── Option1/               # Thư mục chứa các file video .mp4
│   ├── Road_1.mp4
│   ├── Road_2.mp4
│   └── Road_3.mp4
├── output_videos/         # Thư mục chứa video đã overlay kết quả
├── violations.json        # File log vi phạm chung cho tất cả video
├── requirements.txt       # Danh sách thư viện cần thiết
└── README.md
```

## Hướng dẫn sử dụng

1. **vẽ zone**
```
draw_zone.ipynb
```
2. **Cài đặt thư viện**
```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Chạy chương trình**
```bash
python detection.py
```

4. **Kết quả**
- Video đã overlay kết quả sẽ nằm trong thư mục `output_videos/` với tên `overlay_<tên_video>.mp4`.
- File log vi phạm chung sẽ nằm ở `violations.json` với format:
```json
[
  {
    "camera": "Road_1.mp4",
    "timestamp": 12.4,
    "error": "wrong_way",
    "track_id": 5
  },
  ...
]
```

## Tùy chỉnh
- Thay đổi đường dẫn thư mục video, model, output bằng cách sửa các biến đầu file `detection.py`.
- Thêm/sửa zone và hướng hợp lệ trong `polygons.json`.
- Có thể mở rộng lưu thêm các loại vi phạm khác hoặc xuất thêm file CSV nếu cần.

---


