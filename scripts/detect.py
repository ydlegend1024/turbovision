from ultralytics import YOLO
import matplotlib.pyplot as plt


# model = YOLO("yolov8s.pt")

# results = model("test.jpg", conf=0.3, device="cuda")

# annotated_image = results[0].plot()

# plt.figure(figsize=(10, 7))
# plt.imshow(annotated_image[..., ::-1])
# plt.axis("off")

# plt.savefig("output1.png", dpi=200, bbox_inches="tight")

model_players = YOLO("runs/detect/yolov8n_football/weights/best.pt")

# results = model_players.predict(
#     source="soccer-1/test/images",
#     conf=0.3,
#     iou=0.5,
#     device="cuda"
# )


# results = model_players("test.jpg", conf=0.3, device="cuda")
results = model_players("test_1280.png", conf=0.3, device="cuda")

annotated_image = results[0].plot()

plt.figure(figsize=(10, 7))
plt.imshow(annotated_image[..., ::-1])
plt.axis("off")

plt.savefig("output2.png", dpi=200, bbox_inches="tight")