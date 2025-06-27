import cv2
import time
import multiprocessing as mp
from queue import Empty
from utils import ObjectDetector, ObjectTracker, DirectionAnalyzer

# Cấu hình
CAMERA_LIST = {
    "cam1": "rtsp://user:pass@ip1:port/stream1",
    "cam2": "rtsp://user:pass@ip2:port/stream2",
    # Thêm các camera khác...
}
FPS = 15  # FPS mục tiêu cho toàn hệ thống
FRAME_QUEUE_SIZE = 10  # Số frame tối đa lưu trong queue cho mỗi cam

def rtsp_reader(cam_name, rtsp_url, frame_queue, fps):
    cap = cv2.VideoCapture(rtsp_url)
    interval = 1.0 / fps
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] Không đọc được frame, thử lại sau 2s...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        # Đảm bảo queue không bị đầy (bỏ frame cũ nếu cần)
        if frame_queue.qsize() >= FRAME_QUEUE_SIZE:
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put((cam_name, frame, time.time()))
        # Đồng bộ FPS
        elapsed = time.time() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)

def analyzer_worker(frame_queue, result_queue, polygons, model_path="yolo11s.pt"):
    detector = ObjectDetector(model_path, device="cuda:0")
    direction_analyzer = DirectionAnalyzer(arrow_scale=3)
    trackers = {}  # Mỗi cam một tracker riêng
    while True:
        try:
            cam_name, frame, ts = frame_queue.get(timeout=2)
        except Empty:
            continue
        if cam_name not in trackers:
            trackers[cam_name] = ObjectTracker(max_track_length=30)
        tracker = trackers[cam_name]
        # Lấy zone cho cam
        zones = []
        if cam_name in polygons:
            for zone in polygons[cam_name]:
                zones.append({
                    'polygon': [tuple(pt) for pt in zone['points']],
                    'allowed_direction': f"going_{zone['direction']}"
                })
        result = detector.detect_and_track(frame, conf=0.3, iou=0.5)
        violation_list = []
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            tracker.update_tracks(boxes, track_ids)
            for i, track_id in enumerate(track_ids):
                track = tracker.get_track_history(track_id)
                is_wrong, zone = direction_analyzer.check_against_flow(track, zones)
                if is_wrong:
                    violation_list.append({
                        "camera": cam_name,
                        "timestamp": ts,
                        "error": "wrong_way",
                        "track_id": track_id
                    })
            # Overlay kết quả
            direction_analyzer.draw_zones(frame, zones)
            tracker.draw_bboxes_and_ids(frame, boxes, track_ids, classes, detector)
            direction_analyzer.draw_direction_arrows(frame, track_ids, tracker, zones)
        # Đưa kết quả ra queue
        result_queue.put((cam_name, frame, ts, violation_list))

def display_and_log(result_queue, save_video=True, log_file="violations.json"):
    writers = {}
    violation_log = []
    while True:
        try:
            cam_name, frame, ts, violations = result_queue.get(timeout=5)
        except Empty:
            continue
        # Hiển thị
        cv2.imshow(f"Overlay {cam_name}", frame)
        if save_video:
            if cam_name not in writers:
                h, w = frame.shape[:2]
                writers[cam_name] = cv2.VideoWriter(f"overlay_{cam_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
            writers[cam_name].write(frame)
        # Lưu log vi phạm
        violation_log.extend(violations)
        # Nhấn q để thoát tất cả
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Lưu log khi kết thúc
    with open(log_file, "w") as f:
        import json
        json.dump(violation_log, f, indent=4)
    for w in writers.values():
        w.release()
    cv2.destroyAllWindows()

def main():
    # Đọc polygons
    import json
    with open('polygons.json', 'r') as f:
        polygons = json.load(f)
    # Tạo queue chung cho frame và kết quả
    frame_queue = mp.Queue(maxsize=FRAME_QUEUE_SIZE * len(CAMERA_LIST))
    result_queue = mp.Queue()
    # Tạo process đọc RTSP cho từng cam
    readers = []
    for cam_name, rtsp_url in CAMERA_LIST.items():
        p = mp.Process(target=rtsp_reader, args=(cam_name, rtsp_url, frame_queue, FPS))
        p.start()
        readers.append(p)
    # Tạo process phân tích (có thể mở rộng pool)
    analyzer = mp.Process(target=analyzer_worker, args=(frame_queue, result_queue, polygons))
    analyzer.start()
    # Hiển thị và log
    display_and_log(result_queue)
    # Kết thúc
    for p in readers:
        p.terminate()
    analyzer.terminate()

if __name__ == "__main__":
    main()