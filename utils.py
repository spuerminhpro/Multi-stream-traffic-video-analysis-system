import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class ObjectDetector:
    """Module phát hiện đối tượng sử dụng YOLO"""
    
    def __init__(self, model_path="yolo11s.pt", device="cuda:0"):
        self.model = YOLO(model_path)
        self.device = device
        self.class_names = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
        
    def detect_and_track(self, frame, conf=0.3, iou=0.5):
        """
        Phát hiện và theo dõi đối tượng trong frame
        Returns: result object từ YOLO
        """
        result = self.model.track(
            frame, 
            classes=self.class_names, 
            conf=conf, 
            device=self.device, 
            iou=iou, 
            persist=True
        )[0]
        return result
    
    def count_objects(self, result):
        """
        Đếm số lượng phương tiện và người đi bộ
        Returns: (vehicle_count, person_count)
        """
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        person_class = 0
        
        vehicle_count = 0
        person_count = 0
        
        if result.boxes and result.boxes.is_track:
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for c in cls:
                if c == person_class:
                    person_count += 1
                elif c in vehicle_classes:
                    vehicle_count += 1
                    
        return vehicle_count, person_count
    
    def get_class_name(self, cls_id):
        """Chuyển đổi class ID thành tên"""
        class_map = {
            0: "person",
            1: "bicycle", 
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        return class_map.get(cls_id, str(cls_id))


class ObjectTracker:
    """Module theo dõi đối tượng và quản lý ID"""
    
    def __init__(self, max_track_length=30):
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = max_track_length
        
    def update_tracks(self, boxes, track_ids):
        """
        Cập nhật lịch sử track cho các đối tượng
        Args:
            boxes: numpy array các bounding box (xyxy format)
            track_ids: list các track ID
        """
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(float, box)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            track = self.track_history[track_id]
            track.append((cx, cy))
            
            # Giới hạn độ dài track
            if len(track) > self.max_track_length:
                track.pop(0)
                
    def get_track_history(self, track_id):
        """Lấy lịch sử track của một đối tượng"""
        return self.track_history.get(track_id, [])
    
    def draw_tracks(self, frame, track_ids):
        """Vẽ đường track lên frame"""
        for track_id in track_ids:
            track = self.track_history[track_id]
            if len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
    
    def draw_bboxes_and_ids(self, frame, boxes, track_ids, classes, detector):
        """Vẽ bounding box, ID và tên class lên frame"""
        for box, track_id, cls_id in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)
            label = detector.get_class_name(cls_id)
            color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ ID và tên class
            cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


class DirectionAnalyzer:
    """Module phân tích hướng di chuyển của xe"""
    
    def __init__(self, arrow_scale=3):
        self.arrow_buffer = {}
        self.arrow_scale = arrow_scale
        
    def calculate_direction_arrow(self, track, track_id):
        """
        Tính toán và trả về mũi tên hướng di chuyển
        Returns: (start_point, end_point) hoặc None
        """
        if len(track) < 2:
            return None
            
        # Nếu track đạt độ dài tối đa, lưu mũi tên vào buffer
        if len(track) == 30:
            cx, cy = track[-1]
            ox, oy = track[0]
            dx, dy = cx - ox, cy - oy
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                scale = (length / self.arrow_scale) / length
                arrow_dx = dx * scale
                arrow_dy = dy * scale
                start_point = (int(cx), int(cy))
                end_point = (int(cx + arrow_dx), int(cy + arrow_dy))
                self.arrow_buffer[track_id] = (start_point, end_point)
                return start_point, end_point
                
        # Nếu track đã vượt quá độ dài tối đa, dùng mũi tên đã lưu
        elif len(track) > 30 and track_id in self.arrow_buffer:
            return self.arrow_buffer[track_id]
            
        # Track chưa đủ dài, tính mũi tên động
        else:
            cx, cy = track[-1]
            ox, oy = track[0]
            dx, dy = cx - ox, cy - oy
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                scale = (length / self.arrow_scale) / length
                arrow_dx = dx * scale
                arrow_dy = dy * scale
                start_point = (int(cx), int(cy))
                end_point = (int(cx + arrow_dx), int(cy + arrow_dy))
                return start_point, end_point
                
        return None
    
    def point_in_polygon(self, point, polygon):
        """Kiểm tra điểm có nằm trong polygon không"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def analyze_movement_direction(self, track):
        """
        Phân tích hướng di chuyển chỉ dựa vào vị trí ox, cx (không xét góc)
        Returns: (direction_tag, angle)
        """
        if len(track) < 2:
            return "stationary", 0
        
        # Tâm hiện tại là điểm cuối track
        cx, cy = track[-1]
        # Tâm cũ nhất là điểm đầu track
        ox, oy = track[0]
        
        dx = cx - ox
        dy = cy - oy
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if cy > oy:
            direction_tag = "going_down"
        elif cy < oy:
            direction_tag = "going_up"
        else:
            direction_tag = "stationary"
        return direction_tag, angle
    
    def check_against_flow(self, track, zones):
        """
        Kiểm tra xe có đi ngược chiều không
        Returns: (is_against_flow, zone_info)
        """
        if len(track) < 2:
            return False, None
            
        current_position = track[-1]
        direction_tag, movement_angle = self.analyze_movement_direction(track)
        
        # Kiểm tra xem xe có nằm trong zone nào không
        for zone in zones:
            if self.point_in_polygon(current_position, zone['polygon']):
                allowed_direction = zone.get('allowed_direction', 'any')
                
                # So sánh hướng di chuyển với hướng cho phép
                if allowed_direction != 'any' and direction_tag != allowed_direction:
                    return True, zone  # Đi ngược chiều
                    
                return False, zone  # Đi đúng chiều
                
        return False, None  # Không nằm trong zone nào
    
    def draw_direction_arrows(self, frame, track_ids, tracker, zones=None):
        """Vẽ mũi tên hướng di chuyển với màu sắc thể hiện đúng/sai chiều"""
        if zones is None:
            zones = []
            
        for track_id in track_ids:
            track = tracker.get_track_history(track_id)
            arrow_points = self.calculate_direction_arrow(track, track_id)
            
            if arrow_points:
                start_point, end_point = arrow_points
                
                # Kiểm tra ngược chiều
                is_against_flow, zone_info = self.check_against_flow(track, zones)
                
                # Chọn màu mũi tên
                if is_against_flow:
                    color = (0, 0, 255)  # Đỏ - đi ngược chiều
                else:
                    color = (0, 255, 0)  # Xanh lá - đi đúng chiều
                
                cv2.arrowedLine(frame, start_point, end_point, 
                               color=color, thickness=2, tipLength=0.3)
                
                # Hiển thị thông tin hướng di chuyển
                direction_tag, angle = self.analyze_movement_direction(track)
                info_text = f"ID:{track_id} {direction_tag}"
                if is_against_flow:
                    info_text += " (WRONG WAY!)"
                    
                cv2.putText(frame, info_text, (start_point[0], start_point[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_zones(self, frame, zones):
        """Vẽ các zones lên frame"""
        for i, zone in enumerate(zones):
            polygon = zone['polygon']
            points = np.array(polygon, np.int32)
            points = points.reshape((-1, 1, 2))
            
            # Vẽ polygon
            cv2.polylines(frame, [points], True, (255, 255, 0), 2)
            
            # Hiển thị thông tin zone
            center_x = sum([p[0] for p in polygon]) // len(polygon)
            center_y = sum([p[1] for p in polygon]) // len(polygon)
            
            zone_text = f"Zone {i+1}: {zone.get('allowed_direction', 'any')}"
            cv2.putText(frame, zone_text, (center_x - 50, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def get_movement_direction(self, track):
        """
        Phân tích hướng di chuyển của đối tượng (method cũ, giữ để tương thích)
        Returns: string mô tả hướng di chuyển
        """
        direction_tag, angle = self.analyze_movement_direction(track)
        return direction_tag

