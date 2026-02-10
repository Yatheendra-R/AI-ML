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







