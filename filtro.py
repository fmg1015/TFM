import os
import pandas as pd
import shutil

# Clases que te interesan
TARGET_CLASSES = {'drone-bee', 'empty-cells', 'nectar', 'wax-sealed-honey-cells', 'worker-bee'}

# Rutas
ORIGINAL_DATASET = 'dataset_rb'
NEW_DATASET = 'dataset_filtrado'
SETS = ['train', 'valid', 'test']

def ensure_dirs():
    for s in SETS:
        os.makedirs(os.path.join(NEW_DATASET, s, 'images'), exist_ok=True)

def process_set(set_name):
    ann_path = os.path.join(ORIGINAL_DATASET, set_name, '_annotations.csv')
    image_folder = os.path.join(ORIGINAL_DATASET, set_name)
    output_image_folder = os.path.join(NEW_DATASET, set_name, 'images')
    output_csv_path = os.path.join(NEW_DATASET, set_name, 'labels.csv')

    df = pd.read_csv(ann_path)

    # Filtrar solo las clases deseadas
    df = df[df['class'].isin(TARGET_CLASSES)]

    if df.empty:
        print(f"⚠️ Sin clases relevantes en {set_name}, saltando...")
        return

    # Agrupar etiquetas multi-label
    grouped = df.groupby('filename')['class'].apply(lambda x: ','.join(sorted(set(x)))).reset_index()

    # Copiar imágenes correspondientes
    filenames = set(grouped['filename'])
    for fname in filenames:
        src = os.path.join(image_folder, fname)
        dst = os.path.join(output_image_folder, fname)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    # Guardar CSV
    grouped.to_csv(output_csv_path, index=False)
    print(f"✅ Procesado '{set_name}': {len(grouped)} imágenes filtradas")

# Ejecutar todo
ensure_dirs()
for s in SETS:
    process_set(s)
