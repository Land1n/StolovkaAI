from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO('yolov8n.pt')

video = cv2.VideoCapture(r'C:\Users\User\Desktop\work\Stolovka_AI\AI\video.mp4')

mask = cv2.imread('mask.png')

while True:
    _, img = video.read()
    mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion, stream=True)
    cv2.rectangle(img,(300,400),(500,600),(0,0,255),5)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h),l=9)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cvzone.putTextRect(img, f'person {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=3)
    cv2.imshow('Result',img)