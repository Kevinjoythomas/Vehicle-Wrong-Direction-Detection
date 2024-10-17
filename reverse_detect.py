from tracker import Tracker
from ultralytics import YOLO
from metrics import generate_color,poly_containment_ratio,calculate_distance, is_wrong_route
import numpy as np
import configparser
import time
import ast
import cv2
from shapely.geometry import Polygon, box

#Configs
config = configparser.ConfigParser()
config.read("./config/config.ini")
show_gui = config['DEFAULT'].getboolean('show_gui')
print("SHOW GUI:",show_gui)
source = config['DEFAULT'].get('source')
print("Source File:",source)
points = config['DEFAULT'].get('coordinates_of_road')
road_corners = ast.literal_eval(f"[{points}]")
prev_distances = {}
polygon = Polygon(road_corners)
points_array = np.array(road_corners, dtype=np.int32)
vehicle_classes = [2, 3, 5, 6, 7]

model = YOLO('./models/yolov10n.pt')

cap = cv2.VideoCapture(source)

# Reference point on the road
reference_point = (18, 604)

# Tracker
tracker = Tracker()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count+=1
    if (frame_count % 2)!=0:
        continue
    results = model(frame)
    detections = []

    for result in results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            score = float(score)
            if class_id in vehicle_classes:  # Assuming 'car' is the label for cars
                car_bbox = [x1, y1, x2, y2]
                detections.append([x1, y1, x2, y2, score])
                
    detections = np.array(detections)

    tracker_outputs = tracker.update(frame,detections)

    for track in tracker.tracks:
        bbox = track.bbox
        track_id = track.track_id
        track_id = int(track_id)
        x1, y1, x2, y2 = bbox
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        ratio = poly_containment_ratio([x1,y1,x2,y2], polygon)
        if ratio <0.7:
            continue
        

        curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        curr_distance = calculate_distance(curr_center, reference_point)
        
        if track_id in prev_distances:
            prev_distance = prev_distances[track_id]
            if is_wrong_route(curr_distance, prev_distance):
                status = "Illegal"  
                color = (0, 0, 255)  # Red
            else:
                status = "Safe"  
                color = (0, 255, 0)  # Green
        else:
            status = "Detecting..."
            color = (255, 255, 0)  # Yellow

        prev_distances[track_id] = curr_distance

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{track_id}: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    cv2.polylines(frame, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
    if(show_gui):
        cv2.imshow("Wrong route Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the escape key
        break

cap.release()
cv2.destroyAllWindows()