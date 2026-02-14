import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles  #getting data sets
#sklearn is simple and efficient tools for predictive data analysis and is built on top of other scientific libraries like NumPy, SciPy, and Matplotlib.
#Sklearn is mostly used for machine learning, PyTorch is designed for deep learning.
import pandas as pd   #for making table
from sklearn.model_selection import train_test_split #to split the data for training and testing  
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
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

#let me Try to improve the model , with just 2 layers and 0.1 it has around 50% accuracy 

class circle_classi_model(nn.Module): 
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=16)
        self.layer2=nn.Linear(in_features=16,out_features=8)
        self.layer3=nn.Linear(in_features=8,out_features=1)
        #out_features of a layer should match with in_layer of the next layer
        """
        #same thing using sequencing
        self.seq=nn.Sequential(
            nn.Linear(in_features=2,out_features=5),
            nn.Linear(in_features=5,out_features=1)
            )
        """
    def forward(self,x):
        return self.layer3(self.layer2(self.layer1(x))) #self.seq(x) -> using Sequential restrict the complete usage of layers in forward 

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

#torch.manual_seed(42)  
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
                            lr=0.01)

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
"""
What Is Binary Cross Entropy (BCE)?

You are solving a binary classification problem: Label = 0 or 1


Your model outputs a number like:
      0.73
      0.12
    -1.45


But labels are: 0 or 1


So we need a way to measure:

"How wrong is the prediction?"   That’s what BCE does.

Convert Model Output to Probability

For binary classification, we want output between:  0 and 1

So we use:   σ(x)=1/(1+e^−x)

This is the Sigmoid function.

It squashes:
    -∞ → 0
    +∞ → 1


Sigmoid function: It converts: (-∞, +∞) → (0, 1)

Logit	Sigmoid Output
-5	0.0067
0	0.5
2	0.88
5	0.993

Now the output behaves like probability.

Example:

    Raw Output	After Sigmoid
    -5	                 0.006
    0	                 0.5
    3	                 0.95


Compute Binary Cross Entropy Formula

The BCE formula is: Loss=−[ylog(p)+(1−y)log(1−p)]

Where:

    y = true label (0 or 1)
    p = predicted probability

Example 1: Correct Prediction

    True label = 1
    Predicted probability = 0.95

    Loss becomes small.

Example 2: Wrong Prediction

    True label = 1
    Predicted probability = 0.05

    Loss becomes large.

So BCE punishes confident wrong predictions heavily.

Difference Between BCELoss and BCEWithLogitsLoss

    nn.BCELoss()
        Requires input to already be probabilities.
        So you must manually do:
            output = torch.sigmoid(model(x))
            loss = BCELoss(output, y)
        If you forget sigmoid → ❌ Wrong results.

    nn.BCEWithLogitsLoss()

        This does BOTH:
            Sigmoid
            BCE
        Internally it computes: BCE(sigmoid(x),y)

        So you just do: loss = BCEWithLogitsLoss(raw_output, y)
        No need to apply sigmoid manually.

        loss_fn = nn.BCEWithLogitsLoss()
        This means: Model will output raw numbers (called logits)
        Loss function will convert them internally

 Why Is BCEWithLogitsLoss More Stable?

    This is about numerical stability.
    When numbers become very large or very small:
        exp(1000)
        exp(-1000)
    Computers can overflow or underflow.
    BCEWithLogitsLoss combines sigmoid + BCE into one mathematical expression that avoids those extreme computations.

    So:
        More stable
        Less floating point error
        Better gradients

"""
"""
We Directly Use Raw Outputs

Let’s say: logit = 3.5


Is that a probability?❌ No.

Because probability must be: 0 ≤ p ≤ 1

But logits can be:

    -10
    50
    -200
    0.0001


They are just linear combinations of inputs:  z=Wx+b

They have no probabilistic interpretation.

So we Convert to Probability

Because in classification, we want to answer:  “How confident is the model that this belongs to class 1?”

Probability gives meaning:

Probability	Meaning
0.99    	     Very confident class 1
0.50	    Unsure
0.01	     Very confident class 0

Raw logits don’t give intuitive meaning.


Model Prefer Working With Logits

Because: Linear layers naturally output unbounded values.
Gradients behave better with logits.
Numerical stability is better.

If we forced model to output 0–1 directly: Training would be unstable.
"""


"""
What Does “Normalized” Mean?

In simple words: Normalized = scaled into a specific fixed range.

For probabilities, the rules are:

Each value must be between 0 and 1

(For multi-class) all probabilities must sum to 1

Example of normalized probabilities:

[0.7, 0.2, 0.1]


They:

Are between 0 and 1

Sum to 1

That’s normalized.

🔹 What Does “Unnormalized” Mean?

Unnormalized means:

The numbers are raw scores with no restriction.

Example logits:

[ 2.4, -1.7, 0.8 ]


These:

Can be negative

Can be larger than 1

Do NOT sum to 1

Have no probability meaning

They are just scores.

🔹 Why Neural Networks Output Unnormalized Values

The final layer is usually:

z=Wx+b

That is a linear equation.

Linear equations can produce:

-100
0
35.6
-2.1


There is no mechanism forcing outputs into 0–1.

So they are unnormalized.

🔹 Example: Multi-Class Case

Suppose a 3-class classifier outputs logits:

[ 3.2, 1.5, -0.4 ]


These are unnormalized.

To turn them into probabilities, we use Softmax.

Softmax converts them to:

[0.79, 0.18, 0.03]


Now:

All values between 0 and 1

Sum = 1

Interpretable as probabilities

That’s normalization.

Why We Keep Logits Unnormalized During Training

Because:

Linear layers naturally produce unbounded numbers.

Gradients behave better.

Numerically more stable.

Loss functions like BCEWithLogitsLoss expect logits.
"""
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal , and sums, .item() converts outputed (torch) scalar value to normal(python) int 
    acc = (correct / len(y_pred)) * 100 
    return acc
"""
| Component         | What It Does                             |
| ----------------- | ---------------------------------------- |
| Sigmoid           | Converts raw output → probability        |
| BCE               | Measures error for binary classification |
| BCEWithLogitsLoss | Sigmoid + BCE combined safely            |
| Optimizer         | Updates weights                          |
| Learning rate     | Controls step size                       |
| Accuracy          | Measures percentage correct              |

"""
#Train a model
#Going from raw model outputs to predicted labels (logits -> prediction probabilities -> prediction labels)
#for now just going to do for 10 datas

CCM.eval()
with torch.inference_mode():
    Y_pred_no_learn_1=CCM(X_train)[:10]  #raw data / logits  datas b/w -inf --  inf

print("\nLogits or raw data(only first 10  datas):\n")
print(Y_pred_no_learn_1)
Y_pred_prob_no_learn1=torch.sigmoid(Y_pred_no_learn_1)  #converted logits to prediction probabilities , datas b/w 0 -- +1
print("\nconverted logits to prediction probabilities:\n")
print(Y_pred_prob_no_learn1)

"""
If y_pred_probs >= 0.5, y=1 (class 1)
If y_pred_probs < 0.5, y=0 (class 0)
"""
#prediction labels
Y_pred_lab_no_learn1=torch.round(Y_pred_prob_no_learn1)   #to make Y_pred_prob_no_learn1  to look  like Y_train  , so that i can compare it 
print("\nprediction probabilities to prediction labels:\n")
print(Y_pred_lab_no_learn1)
print("Shape: ",Y_pred_lab_no_learn1.shape)
print("Dim: ",Y_pred_lab_no_learn1.dim())
#Y_pred_lab_no_learn1 = (Y_pred_prob_no_learn1 >= 0.5).float()  custom round

#Y_train data
print("\nactual First 10 datas of Y train: ")
print(Y_train[:10])
print("Shape: ",Y_train.shape)
print("Dim: ",Y_train.dim())

#shape of Y_train and Y_pred_lab_no_learn1 does not match so squeeze Y_pred_lab_no_learn1
print("\nSqueezeing Y_pred_lab_no_learn1 to remove 1-dim to match the shape of Y_train:")
Y_pred_lab_no_learn1=Y_pred_lab_no_learn1.squeeze()
print(Y_pred_lab_no_learn1)
print("Shape: ",Y_pred_lab_no_learn1.shape)
print("Dim: ",Y_pred_lab_no_learn1.dim())

#logits -> prediction probabilities -> prediction labels ; This can done in one line
CCM.eval()
with torch.inference_mode():
    Y_pred_label_no_learn1=torch.round((torch.sigmoid((CCM(X_train).squeeze()))))

#sigmoid 1/(1+e^-x)
print("Same thing but using single line: \n",Y_pred_label_no_learn1[:10])


#Training loop
#torch.manual_seed(42)
epochs=200
epoch_cnt=[]
Y_test_loss_arr=[]
Y_train_loss_arr=[]
Y_test_acc_arr=[]
Y_train_acc_arr=[]
# Create a loss function
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=CCM.parameters(), 
                            lr=0.1)
for epoch in range(epochs):
    CCM.train()  #seting training mode
    Y_logits=CCM(X_train).squeeze()  #raw data
    Y_train_pred=torch.round((torch.sigmoid(Y_logits)))
    Y_train_loss=loss_fn(Y_logits,Y_train) ## Using nn.BCEWithLogitsLoss works with raw logits, it has built in sigmoid
    """
    if just nn.BCELoss() used (it does not contain built in sigmoid) so,
    Y_logits=CCM(X_train).squeeze()  #raw data
    Y_train_pred=torch.sigmoid(Y_logits)  so manual sigmoid(making raw data into prob range[0,1])
    Y_train_loss=loss_fn(Y_train_pred,Y_train)   #never put rounded into loss_fn, because learning will be hard
    without rounding it learns better , because it has precision
    """
    acc_train=accuracy_fn(Y_train,Y_train_pred)  #send rounded value as parameter for checking for accuracy

    optimizer.zero_grad()
    Y_train_loss.backward()

    optimizer.step()

    CCM.eval() 
    with torch.inference_mode():
        Y_logits_test=CCM(X_test).squeeze()  #raw data
        Y_test_pred=torch.round((torch.sigmoid(Y_logits_test)))
        Y_test_loss=loss_fn(Y_logits_test,Y_test) ## Using nn.BCEWithLogitsLoss works with raw logits, it has built in sigmoid
        acc_test=accuracy_fn(Y_test,Y_test_pred)  #send rounded value as parameter for checking for accuracy

    if (epoch%10==0):
        print(f"epoch: {epoch} |Train Loss: {Y_train_loss}  | Test Loss: {Y_test_loss} | | Train accuracy: {acc_train}% |Test accuracy: {acc_test}%")
        epoch_cnt.append(epoch)
        Y_test_loss_arr.append(Y_test_loss.detach().numpy())   #to remove grad
        Y_train_loss_arr.append(Y_train_loss.detach().numpy())
        Y_test_acc_arr.append(acc_test)
        Y_train_acc_arr.append(acc_train)

plt.figure(figsize=(8,5))
plt.plot(epoch_cnt,Y_test_loss_arr,label="Test loss")
plt.plot(epoch_cnt,Y_train_loss_arr,label="Train loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epoch_cnt,Y_test_acc_arr,label="Test Acc")
plt.plot(epoch_cnt,Y_train_acc_arr,label="Train Acc")
plt.title("Training and test ACC curves")
plt.ylabel("ACC")
plt.xlabel("Epochs")
plt.legend()
plt.grid(True)
plt.show()

from helper_functions import plot_predictions, plot_decision_boundary   #using the help from other files 
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(CCM, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(CCM, X_test, Y_test)
plt.show()

#conclusion, No it is not working increasing epochs, or layes or training layers ,  There is a missing block - here it is missing non-linearity , this works for linear data types but here data(circle) type is not linear 
