from ultralytics import YOLO
import cv2
import random

def process_video_with_traking(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    classList=['Staff','Buyer','Visitor','Teacher','Student']
    if not cap.isOpened():
        raise Exception('Error: not open video file')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model.track(frame,verbose=False)

        # fps = int(cap.get(cv2.CAP_PROP_FPS))


        if result[0].boxes.id !=  None:
            boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = result[0].boxes.id.cpu().numpy().astype(int)
            cls = result[0].boxes.cls.cpu().numpy().astype(int)
            for box,id in zip(boxes,ids):
                random.seed(int(id))
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,2)
                cv2.putText(
                    frame,
                    f"id {id} {classList[cls[0]]}",
                    (box[0],box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2,
                )
        
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

model = YOLO("StolovkaAI_v1n.pt")

process_video_with_traking(r'C:\Users\User\Desktop\work\Stolovka_AI\AI\video_test\test1.mp4')