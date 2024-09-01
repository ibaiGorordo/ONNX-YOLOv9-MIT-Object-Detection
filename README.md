# ONNX YOLOv9 MIT Object Detection
 Python scripts performing object detection using the YOLOv9 MIT model in ONNX.
 
![!ONNX YOLOv9 Object Detection](https://github.com/user-attachments/assets/a4237b6e-53f1-4c51-be3a-bd8369c1991c)

> [!CAUTION]
> I skipped adding the pad to the input image when resizing, which might affect the accuracy of the model if the input image has a different aspect ratio compared to the input size of the model. Always try to get an input size with a ratio close to the input images you will use.

## Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

## Installation [![PyPI](https://img.shields.io/pypi/v/yolov9-onnx?color=2BAF2B)](https://pypi.org/project/yolov9-onnx/)

```bash
pip install yolov9-onnx
```
Or, clone this repository:
```bash
git clone https://github.com/ibaiGorordo/ONNX-YOLOv9-MIT-Object-Detection.git
cd ONNX-YOLOv9-MIT-Object-Detection
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

## ONNX model
- If the model file is not found in the models directory, it will be downloaded automatically from the [release page](https://github.com/ibaiGorordo/ONNX-YOLOv9-MIT-Object-Detection/releases/tag/0.1.0).
- Or, for exporting the models with a different input size, use the Google Colab notebook to convert the model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KPFC-BLL7i7kQJFYq-1JACFiTzPYyOsC?usp=sharing)
- **Available models**: 
  - **MIT:** v9-s_mit.onnx, v9-m_mit.onnx, v9-c_mit.onnx
  - **Official:** gelan-c.onnx, gelan-e.onnx, yolov9-c.onnx, yolov9-e.onnx

## Original YOLOv9 MIT model
The original YOLOv9 MIT model can be found in this repository: [YOLOv9 MIT Repository](https://github.com/WongKinYiu/YOLO)
- The License of the models is MIT license: [License](https://github.com/WongKinYiu/YOLO/blob/main/LICENSE)

## Usage
```python
import cv2
from yolov9 import YOLOv9, draw_detections

detector = YOLOv9("v9-c_mit.onnx")

img = cv2.imread("image.jpg")

class_ids, boxes, confidences = detector(img)

combined_img = draw_detections(img, boxes, confidences, class_ids)
cv2.imshow("Detections", combined_img)
cv2.waitKey(0)
```

## Examples

 * **Image inference**:
 ```
 python image_object_detection.py
 ```

 * **Webcam inference**:
 ```
 python webcam_object_detection.py
 ```

 * **Video inference**: https://youtu.be/X_XVkEqgCUM
 ```
 python video_object_detection.py
 ```

https://github.com/user-attachments/assets/71b3ef97-92ef-4ddb-a62c-5e52922a396d

## References:
* YOLOv9 MIT model: https://github.com/WongKinYiu/YOLO
* YOLOv9 model: https://github.com/WongKinYiu/yolov9
