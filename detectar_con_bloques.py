import tensorflow as tf
import numpy as np
import cv2

# Configuración
MODEL_PATH = 'modelo_colmena_multilabel.h5'
IMAGE_PATH = 'test_colmena.jpg'
CLASSES = ['drone-bee', 'empty-cells', 'nectar', 'wax-sealed-honey-cells', 'worker-bee']
IMG_SIZE = (128, 128)
BLOCK_SIZE = 128
STRIDE = 64  # Cuánto se desplaza el bloque (puedes probar con 64 o 32)
THRESHOLD = 0.5

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Leer imagen original
image = cv2.imread(IMAGE_PATH)
image_height, image_width = image.shape[:2]
output_image = image.copy()

# Escanear por bloques
for y in range(0, image_height - BLOCK_SIZE + 1, STRIDE):
    for x in range(0, image_width - BLOCK_SIZE + 1, STRIDE):
        block = image[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
        block_resized = cv2.resize(block, IMG_SIZE)
        block_input = block_resized / 255.0
        block_input = np.expand_dims(block_input, axis=0)

        predictions = model.predict(block_input)[0]

        # Procesar predicciones
        for i, prob in enumerate(predictions):
            if prob > THRESHOLD:
                label = f"{CLASSES[i]} ({prob:.2f})"
                cv2.rectangle(output_image, (x, y), (x + BLOCK_SIZE, y + BLOCK_SIZE), (0, 0, 255), 2)
                cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Mostrar resultado
cv2.imshow("Detección con bloques", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar si lo deseas
cv2.imwrite("resultado_deteccion.jpg", output_image)
