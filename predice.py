import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Definiciones
MODEL_PATH = 'modelo_colmena_multilabel.h5'
IMAGE_PATH = 'test_colmena.jpg'  # Cambia esto por tu imagen
CLASSES = ['drone-bee', 'empty-cells', 'nectar', 'wax-sealed-honey-cells', 'worker-bee']
IMAGE_SIZE = (128, 128)  # Asegúrate de que coincide con el entrenamiento

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Cargar imagen con OpenCV
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen {IMAGE_PATH}")

# Mostrar imagen original (opcional)
cv2.imshow("Imagen original", image_bgr)
cv2.waitKey(1000)  # Espera 1 segundo
cv2.destroyAllWindows()

# Redimensionar y normalizar
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image_rgb, IMAGE_SIZE)
normalized_image = resized_image / 255.0
input_array = np.expand_dims(normalized_image, axis=0)

# Predicción
predictions = model.predict(input_array)[0]

# Mostrar resultados
print("\n📊 Resultados de predicción:")
for label, prob in zip(CLASSES, predictions):
    print(f"{label:<25}: {prob:.2f} {'✅' if prob > 0.5 else ''}")

# Opcional: mostrar en imagen
for i, prob in enumerate(predictions):
    if prob > 0.5:
        cv2.putText(image_bgr, f"{CLASSES[i]} ({prob:.2f})", (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Predicción", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
