import os
import cv2
import yaml

# Leer data.yaml para obtener nombres de clases
yaml_path = "dataset/data.yaml"
with open(yaml_path, 'r') as f:
    data_cfg = yaml.safe_load(f)

original_class_names = data_cfg['names']

# Clases objetivo
target_classes = ['empty-cells', 'nectar', 'pollen', 'wax-sealed-honey-cells']
target_class_ids = {name: i for i, name in enumerate(target_classes)}

# Carpeta base del dataset original
base_input_dir = "dataset"
# Carpeta destino formateada para TensorFlow
base_output_dir = "data/dataset_tf"
os.makedirs(base_output_dir, exist_ok=True)

# Procesar train, val, test
splits = ['train', 'valid', 'test']
for split in splits:
    img_dir = os.path.join(base_input_dir, split, 'images')
    lbl_dir = './dataset/train/labelTxt'  # ← no 'labels' si no existe
    out_split_dir = os.path.join(base_output_dir, split)

    # Crear carpetas destino por clase
    for cls in target_classes:
        os.makedirs(os.path.join(out_split_dir, cls), exist_ok=True)

    for label_file in os.listdir(lbl_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(lbl_dir, label_file)
        img_path = os.path.join(img_dir, label_file.replace(".txt", ".jpg"))

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                class_name = parts[8]  # el nombre de la clase está al final


                if class_name not in target_classes:
                    continue

                coords = list(map(float, parts[1:]))
                x_coords = coords[0::2]
                y_coords = coords[1::2]

                x_pixels = [int(x * w) for x in x_coords]
                y_pixels = [int(y * h) for y in y_coords]

                xmin, xmax = max(min(x_pixels), 0), min(max(x_pixels), w)
                ymin, ymax = max(min(y_pixels), 0), min(max(y_pixels), h)

                cropped = img[ymin:ymax, xmin:xmax]
                if cropped.size == 0:
                    continue

                class_out_dir = os.path.join(out_split_dir, class_name)
                out_name = f"{label_file.replace('.txt','')}_{i}.jpg"
                cv2.imwrite(os.path.join(class_out_dir, out_name), cropped)

base_output_dir
