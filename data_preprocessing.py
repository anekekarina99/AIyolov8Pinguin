import os
import shutil

def convert_txt_annotations_to_yolo(txt_dir, images_dir, output_dir):
    """
    Mengonversi anotasi TXT ke format YOLO dan menyalin gambar.
    Format YOLO TXT: class_id x_center y_center width height (0-1 normalized).
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "labels")
    images_output_dir = os.path.join(output_dir, "images")

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # Loop untuk semua file TXT
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith(".txt"):
            image_id = os.path.splitext(txt_file)[0]  # Ambil nama file tanpa ekstensi

            # Path input TXT
            txt_path = os.path.join(txt_dir, txt_file)
            label_output_path = os.path.join(labels_dir, txt_file)

            # Copy TXT ke direktori labels
            shutil.copy(txt_path, label_output_path)

            # Copy gambar dengan nama yang sama
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                shutil.copy(image_path, images_output_dir)
            else:
                print(f"Warning: Gambar {image_id}.jpg tidak ditemukan!")

# Convert train and validation annotations
convert_txt_annotations_to_yolo("data/train_annotations", "data/train/train", "data/yolo_train")
convert_txt_annotations_to_yolo("data/valid_annotations", "data/valid/valid", "data/yolo_valid")