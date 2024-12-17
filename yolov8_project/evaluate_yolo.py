from ultralytics import YOLO

# Load model terlatih
model = YOLO('runs/detect/yolov8_custom_training/weights/best.pt')

# Evaluasi pada data validasi
results = model.val(data='data/data.yaml', imgsz=640, batch=8)

# Tampilkan hasil evaluasi
print("Hasil Evaluasi:", results)
