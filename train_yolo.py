from ultralytics import YOLO

# Load model YOLOv8 pre-trained
model = YOLO('yolov8n.pt')  # YOLOv8n: model nano

# Training model
model.train(data='dataset/data.yaml',
            epochs=50,
            imgsz=640,
            batch=8,
            name='yolov8_custom_training',
            workers=4)

# Menyimpan model
model.save("models/yolov8_custom_model.pt")
