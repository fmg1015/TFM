import os
import torch
import cv2
from flask import Flask, render_template, request, send_file
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Cargar modelo
model_path = os.path.abspath("models/best.pt")
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

@app.route("/")
def index():
    return render_template("index.html", result_image=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)

    # Leer imagen con PIL y redimensionar si es necesario
    pil_img = Image.open(image_path)
    if pil_img.width > 1300:
        new_height = int((1300 / pil_img.width) * pil_img.height)
        pil_img = pil_img.resize((1300, new_height))
        pil_img.save(image_path)  # Sobrescribimos la imagen ya redimensionada

    # Predicción con YOLO
    results = model(image_path)
    pred = results.pandas().xyxy[0]
    image = cv2.imread(image_path)

    for _, row in pred.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Convertimos a base64 para renderizar en HTML
    import base64
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return render_template("index.html", result_image=encoded_img)

if __name__ == "__main__":
    app.run(debug=True)
