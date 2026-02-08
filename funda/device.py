# Check for GPU
import torch
torch.cuda.is_available()
# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Count number of devices
print(torch.cuda.device_count())

# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu,tensor_on_gpu.device)


#Moving tensors back to the CPU
#if you want to interact with your tensors with NumPy (NumPy does not leverage the GPU).

#copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)



