import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
print("imp")

train_data=datasets.FashionMNIST(root="data",download=False,train=True,transform=ToTensor())
test_data=datasets.FashionMNIST(root="data",download=False,train=False,transform=ToTensor())
print(f"train_data => len: {len(train_data)}")
print(f"test_data => len: {len(test_data)}")
#print(test_data) speacial object

label_class_names=train_data.classes
print("Number of  Unique items:",label_class_names)
print("Unique items:",len(label_class_names))

image,label=train_data[0]
print(f"Image shape: {image.shape}")
#print(f"Label shape: {label.shape}")  error , label has single scalar value
plt.imshow(image.squeeze())
plt.title(label_class_names[label])
plt.axis(False)
plt.show()

plt.imshow(image.squeeze(),cmap="grey")
plt.title(label_class_names[label])
plt.axis(False)
plt.show()


fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label_class_names[label])
    plt.axis(False);
plt.show()

BATCH_SIZE=32
train_data_load=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data_load=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

print(f"Dataloaders: {train_data_load, test_data_load}") 
print(f"Length of train dataloader: {len(train_data_load)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_data_load)} batches of {BATCH_SIZE}")
train_features_batch, train_labels_batch=next(iter(train_data_load))  #takes first batch
print(train_features_batch.shape)  #torch.Size([32, 1, 28, 28]
print(train_labels_batch.shape) # torch.Size([32])  , 32 items in each batch - each has one label.

#Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer
class Model_MNIST_CNN(nn.Module):
    def __init__(self,input_shape,hidden_shape,output_shape):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.2)
            )

        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=32,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.2)
            )
            

        #works likemulticlassifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=64*7*7,
                      out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128,
                      out_features=output_shape))
        

    
    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        #print(x.shape)  #torch.Size([10, 14, 14])
        x = self.block2(x)
        #print(x.shape) #torch.Size([10, 7, 7])

        x = self.classifier(x)
       # print(x.shape)
        return x

MMC=Model_MNIST_CNN(1,10,len(label_class_names))

"""
x_random_in=torch.rand(size=(1,1,28,28))
print(x_random_in.shape)

torch.Size([10, 14, 14])
torch.Size([10, 7, 7])
used find the shape of in_feautes after flattening

y_pred=MMC(x_random_in)
"""

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=MMC.parameters(), 
                             lr=0.001)



def accuracy(y_true,y_pred):
    return ((torch.eq(y_true,y_pred).sum().item())/(len(y_true)))*100



epochs=10

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-------")

    train_acc=0;    #python var
    train_loss=0;
    #(X, y) Each batch from the DataLoader returns a tuple
    #enumerate() is a Python function that returns both the index and the value when looping.
    MMC.train()
    for batch, (X, y) in enumerate(train_data_load):
        y_pred_logits=MMC(X)
        
        loss=loss_fn(y_pred_logits,y)
        train_pred_loss=torch.softmax(y_pred_logits,dim=1)
        optimizer.zero_grad()                   
        loss.backward()                    
        optimizer.step()
        train_loss += loss.item()           #Do NOT use .item() before loss.backward(),.item() extracts the scalar value as a plain Python float from a 1-element tensor.
        train_acc+=accuracy(y,torch.argmax(train_pred_loss,dim=1))
        #Mixing Python numbers and tensors is not allowed directly in older PyTorch versions or can create subtle bugs.
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_data)} samples")

    train_acc=train_acc/len(train_data_load)
    train_loss=train_loss/len(train_data_load)

    MMC.eval()
    test_acc=0;  
    test_loss=0;
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_data_load):
            y_pred_logits=MMC(X)
            loss = loss_fn(y_pred_logits, y)

            test_loss+=loss.item()
            test_pred_loss=torch.softmax(y_pred_logits,dim=1)
            test_acc+=accuracy(y,torch.argmax(test_pred_loss,dim=1))

            

        test_acc=test_acc/len(test_data_load)
        test_loss=test_loss/len(test_data_load)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%\n")
"""
Changing from 10 → 32 in first block

In your CNN, hidden_shape = 10 in the first block means 10 filters are used in the first Conv2d layer.
That’s very low: only 10 different “perspectives” or features of the image are being learned.

Why 32 is better:

Each filter learns one type of feature: edges, lines, corners, etc.
With only 10 filters, the network sees very limited types of features → accuracy suffers.
With 32 filters, the network can capture more diverse features at the first layer → better representation → better accuracy.


Changing from 32 → 64 in the second block

After the first block, your image has shape: (batch_size, 32, H, W) (32 channels = 32 feature maps).
Second block sees these 32 channels and now learns higher-level features by combining them.

Why increase channels here:

Deeper layers don’t see the raw image, they see features from previous layers.
To learn more complex combinations of features, we need more filters → more output channels.
So 64 filters in block2 means the network can combine the 32 low-level features into 64 more complex patterns.

Analogy:

Block1: sees edges and corners → 32 “basic sketches”
Block2: combines sketches → 64 “intermediate patterns” like shapes, textures, etc.

Key Idea

First block: capture basic, low-level features → moderate number of channels (32)
Second block: capture more complex, high-level features → increase channels (64, 128…)
Increasing channels deeper = more expressive power, but also more memory and computation.




Adding dropout prevents overfitting:
nn.Dropout(p=0.2)  # randomly zero 20% of activations
Add after ReLU or before classifier.
Helps generalization → better test accuracy.

Adjust Learning Rate & Optimizer

LR = 0.01 worked, but using Adam optimizer can converge faster and better:
optimizer = torch.optim.Adam(MMC.parameters(), lr=0.001)
Adam automatically adjusts learning rate for each parameter → smoother training.


Batch Normalization 
nn.BatchNorm2d(num_features=32)
Normalizes activations → stabilizes training → helps network learn faster → boosts accuracy.
Batch Normalization (BatchNorm)

What it is:
nn.BatchNorm2d(num_features) normalizes the outputs of a layer across the batch dimension, keeping the mean close to 0 and standard deviation close to 1.
num_features = number of channels in the output of the convolutional layer.
For example, if Conv2d outputs 32 channels → nn.BatchNorm2d(32).
BatchNorm normalizes each channel separately.

How it works:
For a batch of activations 
x from one channel:
# For a batch of activations x from one channel:
# Compute mean
mean = (1 / N) * sum(x_i for i in range(1, N+1))
# Compute variance
variance = (1 / N) * sum((x_i - mean)**2 for i in range(1, N+1))
# Normalize->Normalize each activation to have mean 0 and variance 1:
x_hat_i = (x_i - mean) / sqrt(variance + epsilon)
# Scale and shift (learnable parameters)
y_i = gamma * x_hat_i + beta

​N = number of elements in the batch
epsilon (ϵ) = small constant to avoid division by zero
gamma (γ) and beta (β) are learnable parameters
x_hat_i x^(not power it is a cap)i
γ and  β allow the network to learn the optimal scale and shift.

Why it helps:
Stabilizes the distribution of activations → prevents “internal covariate shift.”
Allows higher learning rates without divergence.

Speeds up training → fewer epochs needed.
Often improves final accuracy because the network generalizes better.

Where to place it:
After a convolution, before or after ReLU.
Typically: Conv2d → BatchNorm2d → ReLU.



Dropout

What it is:

nn.Dropout(p=0.2) randomly sets 20% of activations to zero during training.
Helps prevent overfitting, especially when the network has many parameters.
Why it helps:

Forces the network to not rely too heavily on any single neuron.
Each forward pass uses a slightly different network (random subset of neurons).
Acts like an ensemble of many small networks → better generalization.

Where to place it:

After ReLU (activation) in convolutional blocks, or
Before the fully connected layer (classifier).

Intuition:

Think of neurons as team members. If some are randomly “on break” (dropped out), others learn to do the work. This prevents any one neuron from becoming too dominant.


Optimizer & Learning Rate (Adam vs SGD)
SGD (Stochastic Gradient Descent)

Updates weights using gradients: w=w−lr⋅gradient

Simple, but fixed learning rate → may converge slowly or get stuck in local minima.

Adam (Adaptive Moment Estimation)

Combines two ideas:

Momentum → keeps direction from previous updates (like rolling downhill smoothly).
Adaptive learning rate per parameter → each weight gets its own learning rate depending on its past gradients.


Formula (simplified):

Compute running average of gradients (m) and squared gradients (v):
m_t = β1 * m_(t-1) + (1 - β1) * g_t
v_t = β2 * v_(t-1) + (1 - β2) * g_t^2

Update weights:
w = w - lr * (m_t / (sqrt(v_t) + ϵ))


g_t → gradient at time step t
m_t → running average of gradients (momentum)
v_t → running average of squared gradients (RMS)
β1, β2 → decay rates (commonly 0.9, 0.999)
ϵ → small number to avoid division by zero (like 1e-8)
lr → learning rate


Why it helps:

Automatically adapts learning rates → smoother training.
Faster convergence than SGD → fewer epochs to reach high accuracy.
Often better results on complex datasets or deeper networks.
"""



