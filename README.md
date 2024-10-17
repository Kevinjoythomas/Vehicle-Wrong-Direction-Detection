## 🚗  Vehicle Wrong Direction Detection
This project aims to detect vehicles moving in the wrong direction on roads using Computer Vision techniques. It leverages object detection models and tracking algorithms to monitor vehicle movements and raise alerts when a vehicle violates the designated direction of traffic. This system can be deployed on highways, intersections, and restricted zones to improve road safety and traffic management.

![](https://github.com/Kevinjoythomas/Vehicle-Wrong-Direction-Detection/blob/main/img.png)
### 🛠️ Features
Object Detection: Uses YOLOv8 to detect vehicles in real-time.
Vehicle Tracking: Implements Deep SORT for consistent tracking across frames.
Wrong Direction Detection: Compares vehicle trajectories with predefined allowed directions.
Alert System: Logs violations and stores video snippets for review.
Scalable Deployment: Can be integrated into edge devices (like Raspberry Pi) or cloud systems.

### 🚀 How It Works
Detection: The system uses a YOLO-based object detector to identify vehicles in the input video.
Tracking: Each vehicle is assigned a unique ID using Deep SORT to track its path over time.
Direction Check: The direction of vehicle movement is compared against the allowed direction using trajectory analysis.
Alert Generation: If a vehicle moves in the opposite direction, an alert is triggered, and the event is logged.
