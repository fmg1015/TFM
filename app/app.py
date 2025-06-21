import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from collections import Counter, defaultdict

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
MODEL_PATH = 'modelo_colmena_5clases.h5'
CLASSES = ['drone-bee', 'empty-cells', 'nectar', 'wax-sealed-honey-cells', 'worker-bee']
IMG_SIZE = (128, 128)
BLOCK_SIZE = 18
STRIDE = 18
THRESHOLD = 0.5
MAX_WIDTH = 600

# Colores BGR por clase
CLASS_COLORS = {
    'drone-bee': (0, 0, 255),
    'empty-cells': (0, 255, 0),
    'nectar': (255, 0, 0),
    'wax-sealed-honey-cells': (0, 255, 255),
    'worker-bee': (255, 0, 255)
}

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

def merge_boxes(boxes, proximity=10):
    merged = []
    used = [False] * len(boxes)

    for i, box in enumerate(boxes):
        if used[i]:
            continue
        x1, y1, x2, y2 = box
        group = [box]
        used[i] = True

        for j, other in enumerate(boxes):
            if used[j]:
                continue
            ox1, oy1, ox2, oy2 = other
            if abs(x1 - ox1) < proximity and abs(y1 - oy1) < proximity:
                group.append(other)
                used[j] = True

        xs = [b[0] for b in group] + [b[2] for b in group]
        ys = [b[1] for b in group] + [b[3] for b in group]
        merged.append((min(xs), min(ys), max(xs), max(ys)))

    return merged

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return render_template('index.html', result={}, error="No file uploaded")

    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    result_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    file.save(input_path)

    image = cv2.imread(input_path)
    if image is None:
        return render_template('index.html', result={}, error="No se pudo leer la imagen")

    h, w = image.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        image = cv2.resize(image, (MAX_WIDTH, int(h * scale)))

    image_height, image_width = image.shape[:2]
    output_image = image.copy()
    counter = Counter()
    detections_by_class = defaultdict(list)

    for y in range(0, image_height - BLOCK_SIZE + 1, STRIDE):
        for x in range(0, image_width - BLOCK_SIZE + 1, STRIDE):
            block = image[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            block_resized = cv2.resize(block, IMG_SIZE)
            block_input = block_resized / 255.0
            block_input = np.expand_dims(block_input, axis=0)

            predictions = model.predict(block_input, verbose=0)[0]

            for i, prob in enumerate(predictions):
                if prob > THRESHOLD:
                    class_name = CLASSES[i]
                    counter[class_name] += 1
                    detections_by_class[class_name].append((x, y, x + BLOCK_SIZE, y + BLOCK_SIZE))

    for class_name, box_list in detections_by_class.items():
        color = CLASS_COLORS.get(class_name, (0, 0, 0))
        merged_boxes = merge_boxes(box_list, proximity=10)
        for (x1, y1, x2, y2) in merged_boxes:
            overlay = output_image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Relleno
            alpha = 0.5  # 10% opacidad
            output_image = cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0)
           
    cv2.imwrite(result_path, output_image)

    # Convertir colores BGR a HEX para HTML
    def bgr_to_hex(bgr):
        return '#{:02x}{:02x}{:02x}'.format(bgr[2], bgr[1], bgr[0])

    color_map = {name: bgr_to_hex(CLASS_COLORS[name]) for name in CLASSES}

    return render_template('index.html',
                        result=dict(counter),
                        img_path='result.jpg',
                        orig_path='input.jpg',
                        color_map=color_map)

if __name__ == '__main__':
    app.run(debug=True)
