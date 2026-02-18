"""
Why Do We Need CNN?
Your current model(without CNN):  Flatten → Linear → Linear


Problem: It destroys spatial information.
A 28×28 image becomes a flat 784 vector.
The model doesn’t understand that nearby pixels are related.

But in images:

    Nearby pixels are strongly related.
    Edges, shapes, textures matter.
    Spatial structure matters.

CNNs preserve spatial relationships.


What is a Convolutional Neural Network?
    A CNN is a neural network designed specifically for image data.

    Instead of flattening the image, CNNs:
        Apply Convolution layers
        Apply Activation (ReLU)
        Apply Pooling
        Repeat
Flatten only at the end

Use Linear layer for classification

Convolution Layer (Core Idea)
What is a convolution?
    A small filter (kernel) slides over the image and extracts features.

Example:
Input image: 28 × 28
Kernel: 3 × 3
The kernel moves across the image and computes dot products.
    What does it learn?
    Early layers:
        Edges
        Lines
        Corners
    Deeper layers:
        Shapes
        Object parts
        Complex patterns

PyTorch Layer: nn.Conv2d(in_channels, out_channels, kernel_size)
Example: nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

Means:
Input: 1 channel (grayscale)
Output: 32 feature maps
Each feature map learns a different pattern

Feature Maps
If you use: nn.Conv2d(1, 32, 3)
Output becomes: [batch_size, 32, H, W]
Instead of 1 channel, now you have 32 learned feature channels.
Each channel = detector for a specific feature.

Activation Function (ReLU)
After convolution: nn.ReLU()
Why?
    Adds non-linearity.
    Removes negative values.
    Helps model learn complex patterns.

    Without ReLU → model is just linear.

Pooling Layer
Most common: nn.MaxPool2d(kernel_size=2)
What it does:
    Reduces spatial size
    Keeps strongest features
    Makes model faster
    Reduces overfitting

Example: 28×28 → 14×14

Full CNN Flow
For FashionMNIST:
Input: [batch, 1, 28, 28]


Example CNN:
Conv2d(1 → 32)
ReLU
MaxPool
Conv2d(32 → 64)
ReLU
MaxPool
Flatten
Linear


Now the model:

Preserves spatial structure
Learns local patterns
Generalizes better

Why CNN is Better Than Fully Connected?
Fully Connected	CNN
Flattens image	Keeps spatial structure
Many parameters	Fewer parameters
No feature extraction	Automatic feature extraction
Lower accuracy	Higher accuracy

Important Concepts
Local Receptive Field: Each neuron sees only a small region (like 3×3), not entire image.

Weight Sharing:
Same filter applied across whole image.
This reduces parameters drastically.

Translation Invariance
If object moves slightly, CNN still detects it.

Why CNN Performs Better on FashionMNIST

FashionMNIST contains:
    Shoes
    Shirts
    Bags
    Sneakers

These are shape-based objects.

CNN can detect:
    Shoe edges
    Shirt sleeves
    Bag handles

Fully connected cannot efficiently capture these patterns.
"""
"""
nn.Conv2d(1 → 32)
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
Input channels = 1
Output channels = 32
Kernel size = 3×3

What is “in_channels = 1”?

FashionMNIST image shape: [1, 28, 28]

That means:

    1 channel (grayscale)
    Height = 28
    Width = 28

If it was RGB:
    [3, 28, 28]
    Because RGB has 3 channels.

So:
    in_channels=1
    means the layer expects grayscale images.


What is a Kernel (Filter)?

A kernel is a small matrix.
If kernel_size=3: 3 × 3

Example:

[ w1 w2 w3
  w4 w5 w6
  w7 w8 w9 ]


These ws are learnable weights.

What Does the Kernel Do?

It slides over the image.

At each position: Multiply kernel values with image pixels

Add them
Produce one number

That number becomes one pixel in the output feature map.

This operation is called:  Convolution


What is “out_channels = 32”?

It means:
    The layer has 32 different kernels
    Each kernel learns a different feature
    Each kernel produces one feature map

So output becomes: [32, H, W]


Instead of 1 channel → now 32 channels.

Each channel detects something different.

Intuition

Kernel 1 → Detects vertical edges
Kernel 2 → Detects horizontal edges
Kernel 3 → Detects curves
Kernel 4 → Detects corners
...
Kernel 32 → Detects complex pattern

The network learns these automatically.


What is Convolution (Mathematically)?

Imagine:

Input image:

28 × 28


Kernel (filter):

3 × 3


Example kernel:

[ 1  0 -1
  1  0 -1
  1  0 -1 ]


This kernel detects vertical edges.

🔎 Step-by-Step Convolution

Take the top-left 3×3 patch of image:

[ a b c
  d e f
  g h i ]


Multiply element-wise:

(a×1) + (b×0) + (c×-1)
+ (d×1) + (e×0) + (f×-1)
+ (g×1) + (h×0) + (i×-1)


Sum everything → produce one number.

That number becomes the first pixel of the output.

Then move kernel one step right.

Repeat.

How Many Parameters Does Conv2d(1 → 32, 3x3) Have?

Each kernel: 3 × 3 × 1 = 9 weights

Plus 1 bias
So per kernel: 9 + 1 = 10 parameters
We have 32 kernels:32 × 10 = 320 parameters

That’s very small compared to Linear layers.

"""
"""
What is Stride?

Stride tells the kernel:

“How many pixels should I move each time?”

Intuition of Stride

Stride = 1
→ Detailed scanning
→ Larger output
→ More computation

Stride = 2
→ Faster scanning
→ Smaller output
→ Less computation



Why Do We Need Padding?

Without padding: Each convolution shrinks the image.

Example:

28 × 28
Kernel = 3
Stride = 1

Output
(28−3)/1+1=26

Image shrinks: 28 → 26


If we stack many conv layers:
28 → 26 → 24 → 22 → ...


Image becomes tiny quickly.

🔹 What is Padding?
Padding = adding extra pixels around the border.
Usually zeros.

Example:
    Original 5×5:

1 2 3 4 5
...

With padding=1:

0 0 0 0 0 0 0
0 1 2 3 4 5 0
0 ...
0 0 0 0 0 0 0


Now size becomes: 7 × 7


Intuition of Padding

Without padding:

Edges are ignored quickly.
Image shrinks.
Border information lost.

With padding:

Preserve spatial size.
Keep edge information.
Allow deeper networks.


Combining Stride + Padding

Example:

Conv2d(1, 32, kernel_size=3, stride=2, padding=1)


Now:
Kernel = 3
Stride = 2
Padding = 1

Input: 28
Output:
(28−3+2×1)/2+1
Output ≈ 14

So: 28 → 14

Stride=2 halves the size.

Stride controls: How much we shrink by jumping.
Padding controls: Whether we preserve border and size.


What is MaxPooling?

Example:

MaxPool2d(2)


This means:

Take 2×2 block
Keep ONLY the maximum value
Move 2 pixels
Repeat

Example

Input feature map:

[ 1 3
  2 9 ]
After max pooling: 9

It keeps the strongest activation.

Why is this good?
Because convolution outputs:
High values where feature is present
Low values where feature is absent

MaxPooling keeps:
“Is this feature present in this region?”
Instead of caring exactly where.



CNN structure:

Conv(1 → 32)
Conv(32 → 64)
Conv(64 → 128)
Flatten
Linear


Spatial size ↓
Channels ↑
Feature complexity ↑

Why increase channels?

Because deeper layers:

Learn more abstract patterns
Combine lower-level features
Need more representational space
"""
