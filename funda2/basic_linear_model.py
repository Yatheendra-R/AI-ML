import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version

"""
| Topic No. | Topic Title                                           | Contents                                                                                       |
| --------: | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
|         1 | Getting data ready                                    | Data can be almost anything, but to get started we create a simple straight line               |
|         2 | Building a model                                      | Create a model to learn patterns, choose a loss function, optimizer, and build a training loop |
|         3 | Fitting the model to data (Training)                  | Let the model learn patterns from the training data                                            |
|         4 | Making predictions and evaluating a model (Inference) | Compare model predictions with actual testing data                                             |
|         5 | Saving and loading a model                            | Save the model for later use or load it elsewhere                                              |
|         6 | Putting it all together                               | Combine all the above steps into a complete workflow                                           |


what_were_covering = {1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

Machine learning is a game of two parts:

Turn your data, whatever it is, into numbers (a representation).
Pick or build a model to learn the representation as best as possible.


use linear regression to create the data with known parameters (things that can be learned by a model) and
then we'll use PyTorch to see if we can build model to estimate these parameters using gradient descent.
"""
#Y=mx+c
# Create *known* parameters
weight = 0.7   # m
bias = 0.3    # c

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)   #unsqueeze adds dim
print("Tensor X: ",X)
print("Length: ",len(X))
print("Length is 50 becuause 1/0.02",int(1/0.02))
print()

Y = weight * X + bias
print("Tensor Y: ",Y)
print("Length: ",len(Y))
print()
print("For better viewing seing first 10 values: ")

print("X: ",X[:10])
print("Y: ",Y[:10])

#X called features and y called labels


#Split data into training and test sets
"""
| Split          | Purpose                                                                                   | Amount of Total Data | How Often Is It Used? |
| -------------- | ----------------------------------------------------------------------------------------- | -------------------- | --------------------- |
| Training set   | The model learns from this data (like the course materials you study during the semester) | ~60–80%              | Always                |
| Validation set | The model is tuned using this data (like a practice exam before the final)                | ~10–20%              | Often, but not always |
| Testing set    | The model is evaluated on this data to test what it has learned (like the final exam)     | ~10–20%              | Always                |

"""
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
plot_predictions()
