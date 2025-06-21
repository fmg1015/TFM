import torch
print(torch.cuda.is_available())  # Debe imprimir True
print(torch.cuda.device_count())  # Debe mostrar 1 (o más si tienes varias GPUs)
print(torch.cuda.get_device_name(0))  # Debe mostrar "NVIDIA GTX 1060"
