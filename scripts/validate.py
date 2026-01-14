from ultralytics import YOLO
import matplotlib.pyplot as plt


# model = YOLO("yolov8s.pt")

# results = model("test.jpg", conf=0.3, device="cuda")

# annotated_image = results[0].plot()

# plt.figure(figsize=(10, 7))
# plt.imshow(annotated_image[..., ::-1])
# plt.axis("off")

# plt.savefig("output1.png", dpi=200, bbox_inches="tight")

# model = YOLO("runs/detect/yolov8n_football/weights/last.pt")
model = YOLO("player_detect.pt")

model.val(data="soccer-1/data.yaml", device="cuda")

results = model("test.jpg", conf=0.3, device="cuda")
results[0].show()

# annotated_image = results[0].plot()

# plt.figure(figsize=(10, 7))
# plt.imshow(annotated_image[..., ::-1])
# plt.axis("off")

# plt.savefig("output2.png", dpi=200, bbox_inches="tight")