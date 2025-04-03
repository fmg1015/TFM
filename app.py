import os
import torch
import cv2
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

# Cargar el modelo YOLO
model_path = os.path.abspath("models/best.pt")  # Asegúrate de que el modelo esté en esta ruta
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Obtener el archivo
    file = request.files["file"]
    image_path = "temp.jpg"
    file.save(image_path)

    # Hacer la predicción con YOLO
    results = model(image_path)

    # Obtener las predicciones y las cajas
    pred = results.pandas().xyxy[0]  # DataFrame con las predicciones
    image = cv2.imread(image_path)

    # Dibujar las cajas sobre la imagen
    for _, row in pred.iterrows():
        # Coordenadas de la caja: xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']  # Nombre de la clase (por ejemplo, 'persona')
        confidence = row['confidence']  # Confianza de la predicción

        # Dibuja un rectángulo con el color y grosor de la caja
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Color rojo (BGR)

        # Añadir el texto (nombre de la clase y la confianza)
        cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Convertir la imagen a formato RGB y luego a un objeto BytesIO para enviarla
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Devolver la imagen procesada como respuesta HTTP
    return send_file(img_byte_arr, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
