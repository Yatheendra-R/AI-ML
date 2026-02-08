import torch
import numpy as np
import matplotlib.pyplot as plt


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




