from ultralytics import YOLO
import cv2

# Load model YOLOv8 terlatih
model = YOLO('runs/detect/yolov8_custom_training/weights/best.pt')

# Inferensi pada gambar
image_path = "data/valid/valid/image_id_000.jpg"
results = model(image_path)

# Visualisasi hasil deteksi
for result in results:
    im_bgr = result.plot()
    cv2.imshow("YOLOv8 Detection", im_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
