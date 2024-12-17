from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 yang telah dilatih
model = YOLO('runs/detect/yolov8_custom_training/weights/best.pt')  # Path ke model YOLOv8 kamu

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

# Ganti dengan path gambar yang ingin kamu analisis
image_path = 'path/to/your/image.jpg'  # Misalnya 'data/images/your_image.jpg'

# Membaca gambar
frame = cv2.imread(image_path)

# Pastikan gambar berhasil dimuat
if frame is None:
    print(f"Gambar tidak ditemukan di path: {image_path}")
    exit()

# Hitung objek pada gambar
object_counts = count_objects(frame, model)

# Tampilkan hasil counting di gambar
display_text = ", ".join([f"{obj}: {count}" for obj, count in object_counts.items()])
cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Tampilkan gambar dengan hasil counting
cv2.imshow("YOLOv8 Object Counting in Image", frame)

# Tunggu sampai tombol ditekan untuk menutup gambar
cv2.waitKey(0)
cv2.destroyAllWindows()
