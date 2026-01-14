from ultralytics import YOLO
import matplotlib.pyplot as plt


# model = YOLO("yolov8n.pt")
model = YOLO("runs/detect/yolov8n_football/weights/last.pt")

results = model.train(
    data="soccer-1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device="cuda",
    name="yolov8n_football",
    mosaic=False,
)
# model.train(resume=True)

# model_players = YOLO("runs/detect/yolov8n_football/weights/best.pt")
# # model_players = YOLO("football_object_detection.pt")

# results = model_players.predict(
#     source="soccer-1/test/images",
#     conf=0.3,
#     iou=0.5,
#     device="cuda"
# )
