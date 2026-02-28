import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import random

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Generate Circle Data
# -----------------------
theta = np.linspace(0, 2*np.pi, 200)

# True circle (target)
x = np.cos(theta)
y = np.sin(theta)

# ✅ Periodic input encoding (IMPORTANT FIX)
input_features = np.stack(
    (np.sin(theta), np.cos(theta)),
    axis=1
)

X = torch.from_numpy(input_features).float()
Y = torch.stack(
    (
        torch.from_numpy(x).float(),

        torch.from_numpy(y).float()

    ),
    dim=1
)
print(X[:5])
print(X.shape)
print(Y[:5])
# -----------------------
# Shuffle and Split
# -----------------------
dataset = list(zip(X, Y))
random.shuffle(dataset)

X, Y = zip(*dataset)
X = torch.stack(X)
Y = torch.stack(Y)

split_ratio = 0.2
split = int(len(X) * split_ratio)

x_train = X[:-split]
x_test  = X[-split:]

y_train = Y[:-split]
y_test  = Y[-split:]

print("Train size:", x_train.shape)
print("Test size:", x_test.shape)
"""
x_train=X[:80]
y_train=Y[:80]


x_test=X[80:]
y_test=Y[80:]

"""
# -----------------------
# Model
# -----------------------
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = CircleModel()

# -----------------------
# Training Setup
# -----------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 650

# -----------------------
# Training Loop
# -----------------------
for epoch in range(epochs):
    model.train()
    
    pred_train = model(x_train)
    train_loss = loss_fn(pred_train, y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred_test = model(x_test)
        test_loss = loss_fn(pred_test, y_test)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# -----------------------
# Full Prediction
# -----------------------
model.eval()
with torch.no_grad():
    full_pred = model(X)

pred_x = full_pred[:, 0].numpy()
pred_y = full_pred[:, 1].numpy()

# -----------------------
# Plot Result
# -----------------------
plt.figure(figsize=(6,6))
plt.axis('equal')
plt.plot(x, y, label="True Circle", color="green")
plt.scatter(pred_x, pred_y, label="Predicted Circle", color="red", s=10)
plt.legend()
plt.title("Full Circle Prediction")
plt.show()
