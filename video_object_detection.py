from datetime import timedelta
import cv2
from cap_from_youtube import cap_from_youtube

from yolov9 import YOLOv9, draw_detections

# Initialize video
videoUrl = 'https://youtu.be/hmk5cxpAfcw?si=hDOibrI22grM8Aqa'
start_time = timedelta(minutes=4, seconds=0)
cap = cap_from_youtube(videoUrl, start=start_time)

# Initialize object detector
model_path = "models/v9-m_mit.onnx"
detector = YOLOv9(model_path, conf_thres=0.3)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    class_ids, boxes, confidences = detector(frame)

    combined_img = draw_detections(frame, boxes, confidences, class_ids)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break