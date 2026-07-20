import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import CustomLSTM


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# Load Dataset
train_loader, test_loader = get_dataloaders(
    "../data/jena_climate_2009_2016.csv",
    batch_size=64
)


# Model
model = CustomLSTM(
    input_size=1,
    hidden_size=64,
    output_size=12
).to(device)


# Loss Function
criterion = nn.HuberLoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training Settings
epochs = 10

train_losses = []

# ---------------- TRAINING LOOP ---------------- #

os.makedirs("../model_weights", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Loss : {epoch_loss:.6f}")
    print("-" * 40)


# Save Model
torch.save(
    model.state_dict(),
    "../model_weights/custom_lstm.pth"
)

print("Training Complete!")
print("Model Saved!")


# Plot Training Loss
plt.figure(figsize=(8,5))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.grid(True)

plt.savefig("../outputs/training_loss.png")
plt.show()