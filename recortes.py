import os
import pandas as pd
from PIL import Image

# Clases que quieres mantener
clases_deseadas = ['empty-cells', 'nectar','pollen', 'wax-sealed-honey-cells']

# Conjuntos
conjuntos = ['train', 'valid', 'test']

# Directorio base del dataset original
base_dir = 'dataset_seg'
# Directorio base para guardar los recortes
output_base = 'recortes'

for conjunto in conjuntos:
    print(f"Procesando: {conjunto}")

    # Rutas
    ann_path = os.path.join(base_dir, conjunto, '_annotations.csv')
    img_dir = os.path.join(base_dir, conjunto, 'images')

    # Leer CSV
    df = pd.read_csv(ann_path)
    df = df[df['class'].isin(clases_deseadas)]

    for i, row in df.iterrows():
        filename = row['filename']
        clase = row['class']
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        image_path = os.path.join(img_dir, filename)
        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as img:
                cropped = img.crop((xmin, ymin, xmax, ymax))

                # Crear carpeta de salida por conjunto y clase
                output_dir = os.path.join(output_base, conjunto, clase)
                os.makedirs(output_dir, exist_ok=True)

                # Guardar imagen recortada
                base_name = os.path.splitext(filename)[0]
                cropped_filename = f"{base_name}_{i}.jpg"
                cropped.save(os.path.join(output_dir, cropped_filename))
        except Exception as e:
            print(f"❌ Error con {filename}: {e}")

print("✅ Recorte completado.")

