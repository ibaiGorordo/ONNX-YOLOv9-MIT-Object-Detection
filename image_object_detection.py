import cv2
from imread_from_url import imread_from_url

from yolov9 import YOLOv9, draw_detections

# Initialize yolov8 object detector
model_path = "models/v9-c.onnx"
detector = YOLOv9(model_path, conf_thres=0.2)

# Read image
# img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/James_Bowen_and_Bob_the_Street_Cat_busking_in_Covent_Garden_after_the_publication_of_their_new_book._%287510763950%29.jpg/1280px-James_Bowen_and_Bob_the_Street_Cat_busking_in_Covent_Garden_after_the_publication_of_their_new_book._%287510763950%29.jpg"
# img = imread_from_url(img_url)
img = cv2.imread("doc/img/test2.png")

# Detect Objects
class_ids, boxes, confidences = detector(img)

# Draw detections
combined_img = draw_detections(img, boxes, confidences, class_ids)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
