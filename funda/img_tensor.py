import torch

#image tensor
"""
Types of Images
    Grayscale image
    RGB image (single image)
    Batch of images (what models usually take)

"""

#Grayscale image
#(H, W)    H = height (pixels)   ;  W = width (pixels)
gryimag=torch.rand((28, 28))
#print("Grayscale Image: ",gryimag)
print("dim: ",gryimag.ndim)
print("shape: ",gryimag.shape)
print()

#RGB image (single image)
# (C, H, W)  C = 3 → Red, Green, Blue

RGB_img=torch.rand((3, 224, 224))
#print("RGB Image: ",RGB_img)
print("dim: ",RGB_img.ndim)
print("shape: ",RGB_img.shape)
print()

#Batch of images
# (N, C, H, W)   N = batch size
Batch_img=torch.rand(size=(32, 3, 224, 224))
#print("Batch of images: ",Batch_img)
print("dim: ",Batch_img.ndim)
print("shape: ",Batch_img.shape)
