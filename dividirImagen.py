from PIL import Image
import os

# Parámetros
input_path = "file_webp.rf.067a524892efb695b4c266c7693559ea.jpg"
block_size = 64  # tamaño del bloque (en píxeles)
output_dir = "bloques"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Abrir imagen
img = Image.open(input_path)
width, height = img.size

# Recorrer y cortar
count = 0
for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        box = (x, y, x + block_size, y + block_size)
        block = img.crop(box)

        # Evitar bloques incompletos al borde
        if block.size[0] == block_size and block.size[1] == block_size:
            block.save(os.path.join(output_dir, f"bloque_{count}.jpg"))
            count += 1

print(f"✅ {count} bloques guardados en '{output_dir}'")