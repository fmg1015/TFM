
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Rutas
data_dir = 'recortes'
image_size = (128, 128)
batch_size = 32
num_classes = 4
epochs = 30

# Generadores
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1,
                               width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    os.path.join(data_dir, 'valid'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Modelo base
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Capas superiores
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilación
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
history = model.fit(train_data, epochs=epochs, validation_data=val_data)

# Guardar modelo
model.save('/mnt/data/modelo_colmena_4clases_mobilenet.h5')

# Evaluación
test_data.reset()
y_true = np.argmax(test_data.labels, axis=-1)
y_pred = np.argmax(model.predict(test_data), axis=1)
class_labels = list(test_data.class_indices.keys())

print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Graficar precisión
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Precisión por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/mnt/data/precision_plot.png')
plt.show()
