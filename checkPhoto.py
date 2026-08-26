import cv2
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt'))

img_path = "CRITICAL_FAIL_1787117531.jpg"
frame = cv2.imread(img_path)

results = model(frame, verbose=False, conf=0.3)

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    class_id = int(box.cls.cpu().numpy()[0])
    conf = float(box.conf.cpu().numpy()[0])
    cv2.putText(frame, f"{model.names[class_id]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.imwrite("result.png", frame)