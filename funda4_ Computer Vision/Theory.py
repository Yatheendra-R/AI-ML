"""
What is Computer Vision?
Definition : Computer Vision is a field of Artificial Intelligence that enables machines to see, understand, and interpret images or videos, just like humans.

It allows computers to:

    Recognize objects
    Detect faces
    Classify images
    Understand scenes

Computer Vision = Teaching a computer to understand images.

Examples:

    Binary classification → Cat vs Dog
    Multi-class classification → Cat, Dog, Chicken
    Object detection → Where is the car?
    Segmentation → Which pixel belongs to which object?

Extract meaningful information from visual data

What is an Image in Computer Vision?

    A computer does NOT see images like humans.
    It sees an image as a matrix of numbers (pixels).
    It sees numbers CHW, RGB, height and width

    Numbers arranged in a grid->Tensor shape = (Channels, Height, Width)


(Channels, Height, Width) we talked about one image.
grayscale, image size is 28 × 28.-> (1, 28, 28) <- shape
RGB (color image) ,Size = 32 × 32 ->(3, 32, 32)


(Batch_Size, Channels, Height, Width)
NCHW-> Where: N = batch size,C = channels,H = height, W = width
But during training, we don’t pass one image.
We pass a batch of images.

Batch size = 32, Image size = 28×28,Grayscale -> (32, 1, 28, 28)


28 × 28 image
28 pixels in height
28 pixels in width
So 28 = number of pixels.
It does NOT mean :28 cm, 28 inches,Physical size

What is a Pixel?

Pixel = Picture Element
It is:  The smallest unit of an image.

Think of it like a tiny square box.
An image is just: Many tiny squares arranged in a grid
rememeber computer sees an image as a matrix of numbers (pixels).


Each image:
Grayscale
    Size = 28 × 28 pixels

So shape of ONE image: (1,28,28)

Where:
    1 → number of channels
    28 → height
    28 → width
Batch of Images

If we load multiple images at once: (batch_size,channels,height,width)

Example: (1, 1, 28, 28)
Meaning:

    1 image
    1 channel
    28 height
    28 width

If batch size = 32:  (32, 1, 28, 28)


Grayscale Image

A grayscale image contains only intensity values.

Each pixel has a value from:

0→255
0 = Black
255 = White

Between = Shades of Gray

Example: 3×3 Grayscale Image

    0 150 30	​
    50 200 80	​
    100 255 120
	​
Each number represents brightness.

after normalization:

0.0 → Black
1.0 → White


28×28 image means:

28 rows of pixels
28 columns of pixels
Total pixels:28 × 28 = 784 pixels
And each pixel has 1 number (for grayscale).

RGB Example

For RGB:

Each pixel has 3 numbers: (R value, G value, B value)
So a 32×32 RGB image has:
32 × 32 = 1024 pixels
But each pixel has 3 values.
So total numbers:32 × 32 × 3

(3, 128, 128)
It means: 3 channels (RGB)
16,384 pixels
49,152 total stored numbers
128 × 128 = 16,384 pixels
16,384 × 3 = 49,152

flattens a (1, 28, 28) grayscale image
one image tensor (no batch dimension), flattening it gives: (784)

we usually have batch dimension.

If batch size = 1, the shape is:

(1, 1, 28, 28)

After flattening (keeping batch): (1, 784)

Important Understanding

CNN keeps spatial structure: (1, 28, 28)
Linear layer destroys spatial structure: 784 numbers in a row
That’s why CNNs are powerful for images.

Original shape (batch): (32, 1, 28, 28)

Each image → 784 numbers.

Flattening keeps batch dimension but collapses the rest: (32, 784)

Meaning: 32 rows
Each row = one image
Each row has 784 features

This is exactly how a Linear layer expects input: (batch_size, features)

When we flatten: We destroy spatial information.

Example:

In image:

    Pixel (0,0) is next to (0,1)
    Structure matters

After flattening:
    It becomes just a long vector
    Spatial relationships are lost

| **PyTorch Module**            | **What It Does**                                                                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `torchvision`                 | Contains datasets, model architectures, and image transformations commonly used for computer vision problems.                                                                        |
| `torchvision.datasets`        | Provides many example computer vision datasets (image classification, object detection, image captioning, video classification, etc.) and base classes for creating custom datasets. |
| `torchvision.models`          | Includes well-performing and commonly used computer vision model architectures implemented in PyTorch, ready to use for your own problems.                                           |
| `torchvision.transforms`      | Contains common image transformations (turning images into numbers, preprocessing, and data augmentation) before feeding them into models.                                           |
| `torch.utils.data.Dataset`    | Base dataset class in PyTorch for creating your own datasets.                                                                                                                        |
| `torch.utils.data.DataLoader` | Creates a Python iterable over a dataset, supports batching, shuffling, and parallel loading.                                                                                        |

"""
