import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles  #getting data sets
#sklearn is simple and efficient tools for predictive data analysis and is built on top of other scientific libraries like NumPy, SciPy, and Matplotlib.
#Sklearn is mostly used for machine learning, PyTorch is designed for deep learning.
import pandas as pd   #for making table
from sklearn.model_selection import train_test_split #to split the data for training and testing  
n_samples=1000
X,Y=make_circles(n_samples,  
                 noise=0.03,              #with out this it makes perfect circle
                 random_state=42)
"""
X tells the exact position of each dot in 2D space.
 Y tells which group that dot belongs to.
"""
print("First 10 sample of X(features):\n ",X[:10])
print("First 10 sample of Y(Labels):\n ",Y[:10])
#two X values per one y value.
table_circles =pd.DataFrame({"X1":X[:,0],"X2":X[:,1],"Labels":Y})
"""
: means → all rows
0 means → column index 0
"""
#print(table.head(10))  #first 10
print(table_circles.head(10))

print("Count of unique label(y):")
print(table_circles["Labels"].value_counts())

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.RdYlBu)
plt.show()

"""
c = Y
This controls color of each dot.
    If label is 0 → one color
    If label is 1 → another color

cmap = plt.cm.RdYlBu

This means:
    Rd → Red
    Yl → Yellow
    Bu → Blue

It is a color gradient map.
    Label 0 might appear red,
    Label 1 might appear blue.
"""

print("Shape of X features: ",X.shape)
print("Shape of Y Labels: ",Y.shape)
print("Type of X features and Y Lables: ",type(X)," ",type(Y))
#it is in numpy , need to converted into the tensor for learning
X=torch.from_numpy(X).to(dtype=torch.float)  #converted to tensor , type float32
Y=torch.from_numpy(Y).to(dtype=torch.float)
print("Coverted Type of X features and Y Lables(For learning): ",type(X)," ",type(Y))

print("First 10 sample of X(features):\n ",X[:10])
print("First 10 sample of Y(Labels):\n ",Y[:10])


#spliting data for train and test
X_train, X_test,Y_train, Y_test=train_test_split(X,Y,
                                                test_size=0.2, # 80% for train and 20% for train
                                                random_state=42  # make the random split reproducible  like torch.manual_seed(42)
                                                )
print(f"X train length: {len(X_train)}")  #800 is 80% of 1000 n_samples used for training 
print(f"X test length: {len(X_test)}")     #200 is 20% of 1000 n_Samples used for testing 
print(f"Y train length: {len(Y_train)}")
print(f"Y test length: {len(Y_test)}")

#creating a model


class circle_classi_model(nn.Module): 
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=5)
        self.layer2=nn.Linear(in_features=5,out_features=1)
        #out_features of a layer should match with in_layer of the next layer
        """
        #same thing using sequencing
        self.seq=nn.Sequential(
            nn.Linear(in_features=2,out_features=5),
            nn.Linear(in_features=5,out_features=1)
            )
        """
    def forward(self,x):
        return self.layer2(self.layer1(x)) #self.seq(x) -> using Sequential restrict the complete usage of layers in forward 

"""
nn.Linear layers in the constructor capable of handling the input and output shapes of X and y.
Defines a forward() method containing the forward pass computation of the model.


The only major change is what's happening between self.layer_1 and self.layer_2.

self.layer_1 takes 2 input features in_features=2 and produces 5 output features out_features=5.

This is known as having 5 hidden units or neurons.

This layer turns the input data from having 2 features to 5 features.

Why do this?

This allows the model to learn patterns from 5 numbers rather than just 2 numbers, potentially leading to better outputs.

I say potentially because sometimes it doesn't work.

The number of hidden units you can use in neural network layers is a hyperparameter (a value you can set yourself) and there's no set in stone value you have to use.

Generally more is better but there's also such a thing as too much. The amount you choose will depend on your model type and dataset you're working with.


nn.Sequential is fantastic for straight-forward computations, however, as the namespace says, it always runs in sequential order.

So if you'd like something else to happen (rather than just straight-forward sequential computation) you'll want to define your own custom nn.Module subclass.
"""


CCM=circle_classi_model()
print("model: ",CCM)
print("Weight and bias of each layer:\n",CCM.state_dict())
"""
weight = in_feature*out_feature
bias = out_feature

A Linear layer performs this operation:

    y=Wx+b

    Where:

        x = input vector
        W = weight matrix
        b = bias vector
        y = output vector

In your case:
    nn.Linear(2, 5)
    Input size = 2
    Output size = 5

    So:
        Weight matrix W has shape (5 × 2)
        Bias b has shape (5)

Why?  Because we want 5 output numbers, and each output neuron needs 2 weights.

What Is a Neuron Here?

    A neuron is just: Output=w1​x1​+w2​x2​+b
    That’s it.-> It’s just a weighted sum + bias.

    So if you have: nn.Linear(2, 5)

    You have:

    5 neurons
    Each neuron has:
        2 weights
        1 bias

    So total parameters: (2×5)+5=15

The first layer creates 5 different "views" of the data.
Each neuron learns a different weighted combination.
Each neuron learns a different pattern.
So instead of learning from 2 numbers, the next layer learns from 5 learned patterns.
That’s why more hidden units can help.
"""

#first pred without learning
CCM.eval()
with torch.inference_mode():
    Y_pred_no_learn=CCM(X_test)
print("Y pred without learning(200 datas):")
print(Y_pred_no_learn)

print(f"Length of Y_pred_no_learn: {len(Y_pred_no_learn)}  and Shape of Y_pred_no_learn:  {Y_pred_no_learn.shape}")
#a new dim is added to the Y_pred_no_learn, mean a extra square bracket is added to the Y_pred_no_learn to each value making its shape (200,1)
print(f"Length of Y_test: {len(Y_test)}  and Shape of Y_test:  {Y_test.shape}")

print("First 10 sample of Y_pred_no_learn:\n ",Y_pred_no_learn[:10].squeeze()) #removing the dim=1
print("First 10 sample of Y_test:\n ",Y_test[:10])

"""
Why Predictions Look Like Random Numbers?

Because:

Model is untrained

Weights are randomly initialized

It hasn’t learned anything yet
They are just random linear combinations.

Now we go deep into why predictions look random before training.

You saw outputs like:

tensor([[0.0555],
        [0.0169],
        [0.2254],
        ...


And you asked:

Why are these random numbers?

Let’s break it down slowly and clearly.

🔹 Step 1: What Happens When You Create nn.Linear?

When you write:

nn.Linear(2, 5)


PyTorch does NOT set weights to zero.

Instead, it initializes them with random small numbers.

Something like:

W =
[[ 0.32, -0.11],
 [ 0.05,  0.44],
 [-0.29,  0.18],
 ...
]


And biases like:

b = [0.01, -0.03, 0.02, ...]


These are random values drawn from a smart distribution (like Kaiming initialization).

So your model starts as a random mathematical function.

🔹 Step 2: What Does the Model Actually Compute?

Your model is:

layer_2(layer_1(x))


Let’s expand it.

First layer:

ℎ
=
𝑊
1
𝑥
+
𝑏
1
h=W
1
	​

x+b
1
	​


Second layer:

𝑦
=
𝑊
2
ℎ
+
𝑏
2
y=W
2
	​

h+b
2
	​


Combine them:

𝑦
=
𝑊
2
(
𝑊
1
𝑥
+
𝑏
1
)
+
𝑏
2
y=W
2
	​

(W
1
	​

x+b
1
	​

)+b
2
	​


Since all W and b are random…

👉 The whole equation is random.

🔹 Step 3: Example With Real Numbers

Suppose input point:

x = [0.5, -0.3]


Now imagine random weights:

First neuron in layer 1:

(
0.2
∗
0.5
)
+
(
−
0.6
∗
−
0.3
)
+
0.1
(0.2∗0.5)+(−0.6∗−0.3)+0.1

= 0.1 + 0.18 + 0.1
= 0.38

Another neuron:

(
−
0.7
∗
0.5
)
+
(
0.4
∗
−
0.3
)
+
0.05
(−0.7∗0.5)+(0.4∗−0.3)+0.05

= -0.35 - 0.12 + 0.05
= -0.42

These values are completely determined by random weights.

Then layer 2 mixes those again randomly.

Final output becomes some random number like:

0.0555

🔹 Step 4: Why It Doesn’t Match Labels

Your labels are:

0 or 1


But model outputs:

0.0555
0.2254
0.3101


Why?

Because:

The model has not seen any data

No loss calculation happened

No backpropagation happened

No weight update happened

So it's just a random function mapping inputs to outputs.

🔹 Step 5: What Changes After Training?

During training:

Model makes prediction

Compare prediction with real label

Compute loss

Compute gradients

Update weights

Gradually weights stop being random.

They adjust to reduce error.

Then predictions start looking like:

0.92
0.03
0.88
0.01


Now they resemble probabilities.

🔹 Step 6: Important Concept — Random ≠ Meaningless

The outputs look random…

But they are not pure noise.

They are:

👉 Deterministic results of random weights.

If you run the model again with same weights, same input:

You get same output.

It only looks random because weights were randomly initialized.

🔹 Step 7: Why Don’t We Initialize to Zeros?

If we did:

W = 0
b = 0


All neurons would produce identical outputs.

All gradients would be identical.

Model would never learn properly.

So we use random initialization to break symmetry.
"""
"""
Big Insight

Before training:

The neural network is just a random mathematical function.

After training:

It becomes a structured function that separates classes.

 Extra Deep Insight

Because your model has no activation function, it is just a linear model.

So even after training, it can only learn a straight line boundary.

But circle data needs a curved boundary.

That’s why activation functions (ReLU) are critical.

Mental Model

Think of it like this:

Before training:

The network is a baby guessing randomly.

During training:

It gets punished for wrong answers.

After training:

It learns patterns and stops guessing randomly.
"""
#loss 

"""
PyTorch has two binary cross entropy implementations:

    torch.nn.BCELoss() - Creates a loss function that measures the binary cross entropy between the target (label) and input (features).
    torch.nn.BCEWithLogitsLoss() - This is the same as above except it has a sigmoid layer (nn.Sigmoid) built-in (we'll see what this means soon).

Which one should you use?
    The documentation for torch.nn.BCEWithLogitsLoss() states that it's more numerically stable than using torch.nn.BCELoss() after a nn.Sigmoid layer.

    So generally, implementation 2 is a better option. However for advanced usage, you may want to separate the combination of nn.Sigmoid and torch.nn.BCELoss().

Optimizer  torch.optim.SGD() to optimize the model parameters with learning rate 0.1.
"""
"""
"""

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=CCM.parameters(), 
                            lr=0.1)

"""
An evaluation metric can be used to offer another perspective on how your model is going.
If a loss function measures how wrong your model is, I like to think of evaluation metrics as measuring how right it is.
both of these are doing the same thing but evaluation metrics offer a different perspective.
After all, when evaluating your models it's good to look at things from multiple points of view.
There are several evaluation metrics that can be used for classification problems but let's start out with accuracy.

Accuracy can be measured by dividing the total number of correct predictions over the total number of predictions.

For example, a model that makes 99 correct predictions out of 100 will have an accuracy of 99%.

Let's write a function to do so.
"""
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc
