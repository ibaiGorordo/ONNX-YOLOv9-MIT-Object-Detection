import cv2
from cap_from_youtube import cap_from_youtube

from yolov9 import YOLOv9, draw_detections

# # Initialize video
# cap = cv2.VideoCapture("video.avi")
videoUrl = 'https://youtu.be/hmk5cxpAfcw?si=hDOibrI22grM8Aqa'
cap = cap_from_youtube(videoUrl)
start_time = 4*60 # skip first {start_time} seconds
fps = cap.get(cv2.CAP_PROP_FPS)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * fps)

# Initialize object detector
model_path = "models/v9-c_mit.onnx"
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