from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model("modelo_cnn.h5")

# Definir clases del dataset CIFAR-10
CLASSES = ["Avión", "Coche", "Pájaro", "Gato", "Ciervo", "Perro", "Rana", "Caballo", "Barco", "Camión"]

# Ruta principal (Frontend)
@app.route("/")
def index():
    return render_template("index.html")

# Procesar la imagen
def procesar_imagen(imagen):
    imagen = imagen.resize((32, 32))  # Redimensionar
    imagen = np.array(imagen) / 255.0  # Normalizar
    imagen = np.expand_dims(imagen, axis=0)  # Expandir dimensiones
    return imagen

# Ruta para predecir
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se encontró el archivo"}), 400

    archivo = request.files["file"]
    imagen = Image.open(archivo)
    imagen = procesar_imagen(imagen)

    # Predicción
    prediccion = model.predict(imagen)
    clase_predicha = int(np.argmax(prediccion))
    nombre_clase = CLASSES[clase_predicha]

    return jsonify({"clase_predicha": nombre_clase})

if __name__ == "__main__":
    app.run(debug=True)
