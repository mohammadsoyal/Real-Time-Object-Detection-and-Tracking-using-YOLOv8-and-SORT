import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time

# Load YOLOv8 model (make sure yolov8n.pt is in the same directory or path)
model = YOLO("yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])  # class id
        class_name = model.names[cls]  # get class name

        if conf > 0.3:
            detections.append([x1, y1, x2, y2, conf])

    # Convert to numpy array (shape N x 5) or empty array if no detections
    detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    # Update tracker
    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'ID {track_id}'
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow("YOLOv8 + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()