import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# 📁 Parámetros y rutas
modelo_path = "modelo_colmena_transfer.h5"  # O el modelo que hayas entrenado
test_dir = "recortes/test"  # Carpeta con subcarpetas por clase

# 📦 Cargar modelo
model = load_model(modelo_path)

# 🔍 Preparar generador de test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Para que las predicciones coincidan con las verdaderas
)

# 🧠 Hacer predicciones
y_true = []
y_pred = []

test_generator.reset()

for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]
    preds = model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 🔤 Obtener nombres de clases
class_labels = list(test_generator.class_indices.keys())

# 📊 Mostrar métricas por clase
print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred, target_names=class_labels))
