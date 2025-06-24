import os
import cv2

# Configuración
target_classes = {
    'empty-cells': 0,
    'pollen': 1,
    'nectar': 2,
    'wax-sealed-honey-cells': 3,
}

original_class_names = [
    'egg', 'american-foulbrood', 'bee-larvae', 'capped-brood-cells', 'capped-drone-cells',
    'chalk-brood', 'empty-cells', 'multiple-eggs', 'nectar', 'nosema',
    'pollen', 'queen-bee', 'queen-cell', 'small-hive-beetle', 'swarm cells',
    'varroa-mites', 'wax-moth-larva', 'wax-moth-larva-presence', 'wax-sealed-honey-cells', 'worker-bee', 'drone-bee'
]

# Procesar cada partición (train, val, test)
splits = ['train', 'val', 'test']

for split in splits:
    images_dir = f'dataset/images/{split}'
    labels_dir = f'dataset/labels/{split}'
    output_dir = f'dataset_tf/{split}'

    # Crear carpetas por clase
    for class_name in target_classes.keys():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)
        image_path = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # evitar errores por líneas vacías

                class_id = int(parts[0])
                class_name = original_class_names[class_id]

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

                save_dir = os.path.join(output_dir, class_name)
                save_path = os.path.join(save_dir, f"{label_file[:-4]}_{i}.jpg")
                cv2.imwrite(save_path, cropped)
