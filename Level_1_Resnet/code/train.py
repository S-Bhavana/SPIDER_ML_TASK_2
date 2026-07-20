import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader
from model import ResNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Model
model = ResNet().to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Epochs
num_epochs = 10

# Store history
train_loss = []
train_accuracy = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_acc:.2f}%")
    print("-" * 40)

# Save model
os.makedirs("../model_weights", exist_ok=True)

torch.save(model.state_dict(),
           "../model_weights/resnet_cifar10.pth")

print("Training Complete!")
print("Model Saved!")

import matplotlib.pyplot as plt

os.makedirs("../outputs", exist_ok=True)

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs + 1), train_loss, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("../outputs/training_loss.png")
plt.close()

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs + 1), train_accuracy, marker="o")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("../outputs/training_accuracy.png")
plt.close()

print("Training graphs saved successfully!")