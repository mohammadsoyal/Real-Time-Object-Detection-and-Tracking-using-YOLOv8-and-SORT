# Real-Time Object Detection and Tracking using YOLOv8 and SORT

## Project Title

**Real-Time Object Detection and Tracking using YOLOv8 and SORT**

## Objective

To build a system that detects and tracks multiple objects in real-time from webcam or video input using the YOLOv8 deep learning model and the SORT (Simple Online and Realtime Tracking) algorithm.

## Features

- Real-time video feed using webcam
- Object detection using pre-trained YOLOv8 model
- Unique ID assignment and tracking using SORT algorithm
- Bounding boxes and labels displayed on video output
- Efficient performance and easy to extend

---

## Technologies Used

- **Python**
- **OpenCV** (for video capture and display)
- **YOLOv8 (Ultralytics)** (for object detection)
- **SORT** (for tracking objects)
- **NumPy** (for numerical operations)

---

## System Overview

1. **Capture Input**: Read frames from webcam or video file.
2. **Object Detection**: Run YOLOv8 to detect objects in the frame.
3. **Data Formatting**: Convert detection results into a format suitable for tracking.
4. **Object Tracking**: Use SORT to track detected objects frame-by-frame.
5. **Display Results**: Draw bounding boxes and tracking IDs on the frame.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <https://github.com/mohammadsoyal/Real-Time-Object-Detection-and-Tracking-using-YOLOv8-and-SORT>
cd https://github.com/mohammadsoyal/Real-Time-Object-Detection-and-Tracking-using-YOLOv8-and-SORT
```

### 2. Create and Activate Virtual Environment (Windows)

```bash
python -m venv object
object\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have `yolov8n.pt` model downloaded in your project directory.

## File Structure

```
project/
├── main.py                # Main script for detection + tracking
├── sort.py                # SORT algorithm file
├── yolov8n.pt             # YOLOv8 model file
├── requirements.txt       # List of required Python packages
```

## How to Run

1. Make sure your webcam is connected
2. Run the following command:

```bash
python main.py
```

3. Press `q` to quit the window.

---

## Output

- Bounding boxes with labels like "person", "cell phone", etc.
- Real-time video display with tracking IDs

---

## Dependencies (requirements.txt)

```
opencv-python
ultralytics
numpy
filterpy
scikit-image
```

---

## Reference

- [YOLOv8 Documentation - Ultralytics](https://docs.ultralytics.com)
- [SORT Paper](https://arxiv.org/abs/1602.00763)

---

## Future Enhancements

- Add Deep SORT for better tracking with appearance features
- Enable GPU acceleration for higher performance
- Save the output video with bounding boxes

---

##



