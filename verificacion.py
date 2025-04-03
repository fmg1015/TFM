import torch
import torchvision

print(torch.cuda.is_available())  # Debe imprimir True
print(torchvision.__version__)    # Muestra la versión instalada
print(torch.cuda.get_device_name(0))  # Debe mostrar "NVIDIA GeForce GTX 1070"
