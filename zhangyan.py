from ultralytics import YOLO
import cv2
# Load a model

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

video_path = "D:/workspace/mp4/test.mp4"

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
    results = model.predict(source = frame)
    result = results[0]
    anno_frame = result.plot()
    cv2.imshow("V8",anno_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows

# Use the model

# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# save_path = "D:\workspace\mp4"
# results = model.predict(source, stream=True)
# print(results)
