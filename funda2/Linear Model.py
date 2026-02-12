import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

#y=mx+c
weight=0.7
bias=0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)   #unsqueeze adds dim
print("Tensor X: ",X)
print("Length: ",len(X))
print()

Y = weight * X + bias
print("Tensor Y: ",Y)
print("Length: ",len(Y))
print()


# Create train/test split
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
print("nummber of data for traing and testing: ",train_split,"  ",len(X)-train_split)
X_train, y_train = X[:train_split], Y[:train_split]
X_test, y_test = X[train_split:], Y[train_split:]


print("X_train: ",len(X_train))
print("y_train: ",len(y_train))
print("X_test: ",len(X_test))
print("Y_test ",len(y_test))

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
      plt.figure(figsize=(10, 7))

      # Plot training data in blue
      plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
      
      # Plot test data in green
      plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

      if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

      # Show the legend
      plt.legend(prop={"size": 14});
      plt.show()
plot_predictions()

class prediction_linear_model(nn.Module):      ##nn.module is parent of it , inheritance To known more about nn.module refer to file called nn_module
    def __init__(self):  #constructor of prediction_linear_model
        super().__init__()  #calling constructor nn.module (parent of prediction_linear_model)
        self.weight=nn.Parameter(torch.randn(1, # <- start with random weights (this will get adjusted as the model learns)
                                 dtype=torch.float), # <- PyTorch loves float32 by default
                                 requires_grad=True) # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(torch.randn(1, 
                                             dtype=torch.float),
                                 requires_grad=True) 

        # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
        return self.weight* x + self.bias # <- this is the linear regression formula (y = m*x + b)

# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))

PLM=prediction_linear_model()
# Check the nn.Parameter(s) within the nn.Module subclass we created
print(PLM.parameters())
print(list(PLM.parameters()))
#List named parameters 
print(PLM.state_dict())


# Make predictions with model
#This is doing prediction (inference), not training.
with torch.inference_mode(): 
    y_preds = PLM(X_test)         # internal y_preds = PLM.forward(X_test)

"""
calls forward()
uses current weights and bias
computes predictions
y = w·x + b


Inside this block:

Gradient tracking is DISABLED:
No computation graph:
No .grad storage
Faster execution: Skips autograd bookkeeping
Less memory usage: Especially important for big models
Safer inference: Prevents accidental .backward()


By default, PyTorch:

      tracks every operation
      builds a computation graph
      prepares for backpropagation

But during testing / prediction:

      we are NOT calling loss.backward()
      we do NOT want gradients
      we just want numbers
"""
"""
Note: in older PyTorch code you might also see torch.no_grad()
with torch.no_grad():
      y_preds = PLM(X_test)
"""

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")
print(f"required values:\n{y_test}")

plot_predictions(predictions=y_preds)
# predictions look pretty bad...
#our model is just using random parameter values to make predictions. It hasn't even looked at the blue dots to try to predict the green dots.

print("difference b/w y_test-y_preds",y_test-y_preds)


#loss function and optimizer

"""
Function        | What does it do?                                                                                                                    | Where does it live in PyTorch?| Common values
------------|-----------------------------------------------------------------------------|----------------------------|-----------------------------------------------
Loss function| Measures how wrong your model's predictions (e.g., y_preds) are compared to the   | torch.nn                                     | MAE for regression: torch.nn.L1Loss()
                        true labels (e.g., y_test). Lower loss means better performance.                                   |                                                     | Binary classification: torch.nn.BCELoss()

Optimizer      | Decides how the model should update its internal parameters (weights & biases)    | torch.optim                                | SGD: torch.optim.SGD()
                       | in order to minimize the loss function.                                                                              |                                                     | Adam: torch.optim.Adam()

"""


"""
the SGD (stochastic gradient descent) or Adam optimizer.
And the MAE (mean absolute error) loss function for regression problems (predicting a number)
binary cross entropy loss function for classification problems (predicting one thing or another).
"""
"""
Mean absolute error (MAE, in PyTorch: torch.nn.L1Loss) measures the absolute difference between
two points (predictions and labels) and then takes the mean across all examples.
"""

"""
SGD, torch.optim.SGD(params, lr) where:

params is the target model parameters you'd like to optimize (e.g. the weights and bias values we randomly set before).
lr is the learning rate you'd like the optimizer to update the parameters at, higher means the optimizer
will try larger updates (these can sometimes be too large and the optimizer will fail to work),
lower means the optimizer will try smaller updates (these can sometimes be too small and the optimizer
will take too long to find the ideal values).
The learning rate is considered a hyperparameter (because it's set by a machine learning engineer).
Common starting values for the learning rate are 0.01, 0.001, 0.0001,
however, these can also be adjusted over time (this is called learning rate scheduling).
"""
"""
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))
"""

"""
 Loss Functions (the “judge” )

 A loss function tells how bad the model’s prediction is using a single numerical value.

What a loss function really measures:
      Difference between prediction vs truth
      Loss vs Error vs Cost (common confusion)
Types of loss functions:
      Regression losses (e.g., MSE, MAE)
      Classification losses (e.g., Cross-Entropy)

A loss function converts model mistakes into a single number that tells us how bad the prediction is.

      Smaller number → better model
      Larger number → worse model
      0 loss → perfect prediction
Loss function → “How wrong am I?”
Two common regression loss functions (intuition only)
Mean Squared Error (MSE)
      Squares the error
      Large mistakes hurt a LOT,  Squaring blows up large errors, so outliers dominate the loss.
      Sensitive to outliers
      Very smooth → great for optimization
      MSE → torch.nn.MSELoss

Decreasing loss means the model is learning patterns in the data and making predictions closer to the true values.


Mean Absolute Error (MAE)
      Takes absolute value
      Treats all mistakes linearly
      More robust to outliers
      Less smooth (harder to optimize)
      MAE → torch.nn.L1Loss


"""
"""

Optimizers (the “mechanic” 🔧)

minimizing loss isn’t trivial

Gradient Descent intuition: “walking downhill” idea
Learning rate: Too small vs too big (and why models explode)

Popular optimizers:

      SGD
      Momentum
      Adam (and why everyone uses it)

What optimizers actually update: weights, biases, parameters


You should know why Adam works better than plain SGD in most cases.

Loss + Optimizer together (the training loop )

What we’ll connect: How loss → gradients → optimizer → updated weights

loss.backward() 
optimizer.step() 



Picture this 👇

      You’re standing on a hill in thick fog 
      Your goal: reach the lowest point (valley).
      But:

      You can’t see the whole hill
      You only feel the slope under your feet

      You move in the direction of the steepest downward slope.

      Hill height → loss value
      Your position → model parameters (weights & bias)
      Steepest downward direction → negative gradient

Optimizer looks at the gradient and moves weights in the direction that reduces loss fastest.


Gradient = direction + strength

Tells:
      Which way to move
      How steep the loss surface is

If gradient is:

      Big → steep hill
      Small → flat area

One critical knob: Learning Rate

Learning rate decides step size:

      Too big → you overshoot the valley ❌
      Too small → you crawl forever 🐢
      Just right → smooth descent ✅

This is why models sometimes:

      diverge
      oscillate
      converge nicely

Optimizers move parameters in the opposite direction of the gradient to reduce loss.

Gradient → points uphill
Optimizer → walks downhill

"""
"""
Stochastic Gradient Descent (SGD)


Compute loss : Compute gradient of loss w.r.t weights

Update weights like:  new weight = old weight − (learning rate × gradient)


Why plain SGD is sometimes bad :

      Zig-zags in narrow valleys
      Sensitive to learning rate
      Slow on complex surfaces
      Gets noisy with mini-batches



Enter Adam (intuition only)

Adam keeps memory:

      Remembers past gradients (momentum)
      Adapts step size per parameter
      Moves fast on flat areas
      Slows down near minima

That’s why:

      Adam converges faster
      Works well out-of-the-box
      Is the default choice for many models

Connect to PyTorch (names only)

      SGD → torch.optim.SGD
      Adam → torch.optim.Adam

gradient of loss w.r.t weights” even mean?


loss → a number (how bad the model is)
weights → numbers inside the model

w.r.t → “with respect to”

So the question becomes: If I slightly change a weight, how does the loss change?

That change = gradient.

What gradient tells the optimizer

From the above thought:

      If increasing w makes loss go down
      → gradient is negative
      If increasing w makes loss go up
      → gradient is positive

So gradient answers:

      Which direction to move?
      How strongly?

"""
"""
Gradient → tells which way is downhill
Optimizer → moves weights downhill
Momentum → remembers past gradients
Consistent direction → speeds up
Random direction → smooths / stabilizes
"""

#Traing loop
"""
PyTorch Training Loop

Step | Name                | What does it do?
-----|---------------------|---------------------------------------------------------------
1    | Forward pass        | Passes input data through the model to get predictions.
2    | Calculate loss      | Compares predictions with true labels and produces a single
     |                     | value measuring how wrong the model is.
3    | Zero gradients      | Clears previously stored gradients because PyTorch accumulates
     |                     | gradients by default.
4    | Backward pass       | Computes gradients of the loss with respect to each parameter
     |                     | (for parameters with requires_grad = True).
5    | Optimizer step      | Updates model parameters using the computed gradients to reduce
     |                     | the loss.


Number | Step name                         | What does it do?                                                                 | Code example
-------|-----------------------------------|----------------------------------------------------------------------------------|----------------------------
1      | Forward pass                      | The model goes through all of the training data once, performing its forward()   | model(x_train)
       |                                   | function calculations.                                                           |
2      | Calculate the loss                | The model's outputs (predictions) are compared to the ground truth and evaluated | loss = loss_fn(y_pred, y_train)
       |                                   | to see how wrong they are.                                                        |
3      | Zero gradients                    | The optimizer's gradients are set to zero (they are accumulated by default) so   | optimizer.zero_grad()
       |                                   | they can be recalculated for the specific training step.                          |
4      | Perform backpropagation on loss   | Computes the gradient of the loss with respect to every model parameter to be     | loss.backward()
       |                                   | updated (each parameter with requires_grad=True). This is known as                |
       |                                   | backpropagation, hence "backwards".                                               |
5      | Update the optimizer              | Updates the parameters with requires_grad=True with respect to the loss           | optimizer.step()
       | (gradient descent)                | gradients in order to improve them.                                               |


train
forward
loss
zero grad
grad
optim

"""
torch.manual_seed(42)
epochs=1750 #number of times train the model   // 200
#to keep a track of the loss values
train_loss_values=[]
test_loss_values=[]
epoch_count=[]

loss_fn = nn.L1Loss()   # useing MAE
optimizer = torch.optim.SGD(PLM.parameters(), lr=0.001) #optimizer = torch.optim.SGD(PLM.parameters(), lr=0.01)


for epoch in range(epochs):
      #training
      #Put model in training mode (this is the default state of a model)

      PLM.train()
      #disables few neurons so that it should not memorize the data
      """
      """

      # 1. Forward pass on train data using the forward() method inside 
      y_train_pred=PLM(X_train)      #x_train data is sent to the forward means first 40 datas , here it tries to predict y_trian , using randn weights and bais giving y_train_pred
      #This might not be correct because it used rand. value of weight and bias to pred , so we are going to train it
      """
      Forward pass
            Model takes input x
            Produces prediction ŷ
      """

      #Here before teaching the model about correct weight and bias , we need to find the loss,b/w y_train_pred and y_train
      #y_train_pred, values got while sending first 40 x_train into forward , and tried to pred y_train using its random weight and bias
      #y_train, excepted 40 y_train for 40 x_train value, these are the values that y_train_pred should give , we should train model to change weight and bias to give correct value

      #train_loss loss during train data

      train_loss=loss_fn(y_train_pred,y_train)
      """
      Compute loss:
            Compare ŷ with true y    
            Get a single number (loss)
      """



      # 3. Zero grad of the optimizer
      optimizer.zero_grad()
      #grad accumulate, so we need to make it zero
      """
      Zero gradients:
            Clear old gradients before next step
            This repeats every batch / epoch.
      """


      # 4. Loss backwards
      train_loss.backward()
      #it helps to find the direction to move weight and bias
      """
      Backward pass:
            Compute gradients of loss w.r.t weights
            Stored in parameter.grad
      """
      # 5. Progress the optimizer
      optimizer.step()
      #it changes the weight and bias towards less loss,here model started becoming better
      #Now we are going to give the test data, how good it has learnt (pred the correct value of weight and bias)
      """
      Optimizer step:
            Update weights using gradients
            Move downhill
      """
"""
Mental dry-run
setup

One weight: w

Input: x = 2

True value: y = 10

Learning rate: 0.1

Step A: Forward pass

      Assume:

            w = 3
            ŷ = w × x = 6
            
            Model is under-predicting.

Step B: Loss

      Loss is positive and big (prediction far from 10).
      You don’t care about the exact number — only that it’s high.

Step C: Backward pass

      PyTorch figures out:
            Increasing w → increases ŷ
            Increasing ŷ → reduces loss

      So:

            Gradient of loss w.r.t w is negative
            This is the key result.

Step D: Optimizer step

      Optimizer does:
            w_new = w − lr × gradient
            Since gradient is negative:
            Minus (negative) → w increases
            
            So:
                  w goes from 3 → something bigger
                  Prediction gets closer to 10 ✅

Step E: Zero grad
      Gradients cleared so they don’t accumulate accidentally.


"""

      """
      Forward tells me where I am, loss tells me how bad it is, backward tells me which way is down,
      optimizer moves me, zero_grad resets for the next step.
      """

      PLM.eval() 

      with torch.inference_mode():
            test_Y_pred = PLM(X_test)
            test_loss = loss_fn(test_Y_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(train_loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {train_loss} | MAE Test Loss: {test_loss} ")


# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(PLM.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

      
# 1. Set the model in evaluation mode
PLM.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  test_Y_pred = PLM(X_test)
print(test_Y_pred)
print(y_test)
plot_predictions(predictions=test_Y_pred)
      
"""
Why do we need optimizer.zero_grad()?

Key line from your table (this is the hint):

“they are accumulated by default”

What that means exactly

In PyTorch:

Gradients are added, not replaced

Every time you call loss.backward(), PyTorch does:

param.grad += new_gradient


So if you don’t zero them:

Step 1 gradient → stored

Step 2 gradient → added on top

Step 3 → added again
❌ Now gradients are wrong

In one clean sentence (exam + interview safe)

We call optimizer.zero_grad() to clear old gradients because PyTorch accumulates gradients by default, and we only want gradients from the current training step.
"""

"""
STEP 1: Forward pass
What is happening?

You give input data to the model.

The model uses its current weights and biases.

It produces predictions.

Nothing is learned here yet.

Important points

Only calculations, no updates

Uses the model’s forward() logic

Output depends entirely on current parameters

Mental picture

“Given what I know right now, this is my guess.”

STEP 2: Calculate the loss
What is happening?

The model’s predictions are compared with the true labels.

A loss function converts this difference into one number.

This number tells us how bad the prediction is.

Important points

Loss is a single scalar value

Lower loss = better prediction

Different problems → different loss functions

Mental picture

“How wrong was my guess?”

STEP 3: Zero gradients
Why is this step needed?

In PyTorch:

Gradients are accumulated by default

New gradients are added to old ones

If we don’t clear them:

Gradients become incorrect

Updates become wrong

What happens here?

All stored gradients are reset to zero

Prepares for fresh gradient calculation

Mental picture

“Erase the board before writing new answers.”

STEP 4: Backpropagation (backward pass)
What is happening?

PyTorch computes gradients automatically

For each parameter (weight, bias):

How much does changing this affect the loss?

This uses the chain rule, but PyTorch handles it.

Important points

No parameter is updated yet

Gradients are stored in param.grad

Only parameters with requires_grad=True are included

Mental picture

“Which direction should each weight move to reduce loss?”

STEP 5: Optimizer step
What is happening?

The optimizer reads the gradients

Updates the parameters using:

gradient direction

learning rate

(and momentum / adaptive logic if used)

This is where learning actually happens.

Important points

Parameters are modified here

Loss should decrease over time

Different optimizers update differently

Mental picture

“Move the weights downhill.”

ONE FULL CYCLE IN ONE STORY

Forward → make a guess

Loss → check how bad the guess is

Zero grad → clear old memory

Backward → find directions to improve

Step → update knowledge

Repeat this hundreds or thousands of times.
"""
      
