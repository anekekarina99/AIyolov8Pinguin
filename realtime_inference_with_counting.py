from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 yang telah dilatih
model = YOLO('runs/detect/yolov8_custom_training/weights/best.pt')  # Ganti dengan path model yang sesuai

# Fungsi untuk menghitung objek berdasarkan label
def count_objects(frame, model):
    # Deteksi objek pada gambar
    results = model(frame)

    # Dapatkan prediksi label dan confidence
    pred = results[0].pred[0]  # Prediksi kelas dan koordinat objek
    labels = results[0].names  # Nama kelas objek (dari data.yaml)

    # Inisialisasi dictionary untuk counting objek
    object_counts = {name: 0 for name in labels.values()}

    # Loop melalui prediksi dan hitung jumlah objek per label
    for label in pred[:, -1].unique():  # Prediksi label objek
        count = (pred[:, -1] == label).sum().item()  # Hitung jumlah objek dengan label yang sama
        object_counts[labels[int(label.item())]] = count
    
    return object_counts

# Video Capture (gunakan '0' untuk webcam atau path video file)
cap = cv2.VideoCapture(0)  # Gunakan '0' untuk webcam atau ganti dengan path video jika menggunakan file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Hitung objek pada frame
    object_counts = count_objects(frame, model)

    # Tampilkan hasil counting di frame
    display_text = ", ".join([f"{obj}: {count}" for obj, count in object_counts.items()])
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame dengan hasil deteksi dan counting
    cv2.imshow("YOLOv8 Real-Time Detection & Counting", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber daya setelah selesai
cap.release()
cv2.destroyAllWindows()
