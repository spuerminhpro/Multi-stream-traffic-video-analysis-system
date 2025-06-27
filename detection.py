import cv2
import os
import json
import threading
import multiprocessing
from queue import Queue
import numpy as np
import csv
from utils import ObjectDetector, ObjectTracker, DirectionAnalyzer

# Cấu hình
VIDEO_FOLDER = r"D:\Python_project\ORBRO\Option1"
OUTPUT_FOLDER = "output_videos"
FPS = 25
VIOLATION_LOG_FILE = "violations.json"
ANALYSIS_CSV_FILE = "analysis.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Đọc file JSON chứa các polygon
with open('polygons.json', 'r') as f:
    polygons = json.load(f)

video_files = [os.path.join(VIDEO_FOLDER, f) for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]

violation_log = []  # Dùng chung cho tất cả video
violation_log_lock = threading.Lock()

def draw_summary(frame, vehicle_count, person_count):
    """Vẽ bảng tổng hợp số lượng đối tượng"""
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Pedestrians: {person_count}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

def process_video(video_path):
    video_name = os.path.basename(video_path)
    detector = ObjectDetector("yolo11s.pt", device="cuda:0")  # Model riêng cho mỗi thread
    tracker = ObjectTracker(max_track_length=30)
    direction_analyzer = DirectionAnalyzer(arrow_scale=3)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(OUTPUT_FOLDER, f"overlay_{video_name}")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    zones = []
    if video_name in polygons:
        for zone in polygons[video_name]:
            zones.append({
                'polygon': [tuple(pt) for pt in zone['points']],
                'allowed_direction': f"going_{zone['direction']}"
            })
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Đảm bảo FPS ổn định
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        result = detector.detect_and_track(frame, conf=0.3, iou=0.5)
        direction_analyzer.draw_zones(frame, zones)
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            vehicle_count, person_count = detector.count_objects(result)
            draw_summary(frame, vehicle_count, person_count)
            tracker.update_tracks(boxes, track_ids)
            tracker.draw_bboxes_and_ids(frame, boxes, track_ids, classes, detector)
            direction_analyzer.draw_direction_arrows(frame, track_ids, tracker, zones)
            for i, track_id in enumerate(track_ids):
                track = tracker.get_track_history(track_id)
                is_wrong, zone = direction_analyzer.check_against_flow(track, zones)
                if is_wrong:
                    violation_data = {
                        "camera": video_name,
                        "timestamp": frame_count / FPS,
                        "error": "wrong_way",
                        "track_id": track_id
                    }
                    with violation_log_lock:
                        violation_log.append(violation_data)
        out.write(frame)
        frame_count += 1
    cap.release()
    out.release()

def main():
    # Xử lý đa luồng cho từng video
    threads = []
    for video in video_files:
        t = threading.Thread(target=process_video, args=(video,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    # Lưu log vi phạm chung
    with open(VIOLATION_LOG_FILE, "w") as f:
        json.dump(violation_log, f, indent=4)
    print("Xử lý xong tất cả video.")

if __name__ == "__main__":
    main()
