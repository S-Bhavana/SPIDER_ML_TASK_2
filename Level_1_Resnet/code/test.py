import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import test_loader
from model import ResNet

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load model
model = ResNet().to(device)
model.load_state_dict(torch.load("../model_weights/resnet_cifar10.pth", map_location=device))
model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

print("\nClassification Report:\n")
print(classification_report(all_labels, all_predictions, target_names=classes))

cm = confusion_matrix(all_labels, all_predictions)


import os

os.makedirs("../outputs", exist_ok=True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45)

plt.title("Confusion Matrix")

plt.savefig("../outputs/confusion_matrix.png")

plt.show()

print("Confusion matrix saved successfully!")