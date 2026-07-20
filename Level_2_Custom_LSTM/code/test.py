import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import get_dataloaders
from model import CustomLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load test data
_, test_loader = get_dataloaders(
    "../data/jena_climate_2009_2016.csv",
    batch_size=64
)

# Load model
model = CustomLSTM(
    input_size=1,
    hidden_size=64,
    output_size=12
).to(device)

model.load_state_dict(torch.load("../model_weights/custom_lstm.pth", map_location=device))
model.eval()

criterion = nn.HuberLoss()

all_preds = []
all_targets = []

test_loss = 0.0

with torch.no_grad():

    for inputs, targets in test_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.item()

        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

test_loss /= len(test_loader)

mae = mean_absolute_error(all_targets, all_preds)
mse = mean_squared_error(all_targets, all_preds)

print("\nHuber Loss :", round(test_loss, 6))
print("MAE        :", round(mae, 6))
print("MSE        :", round(mse, 6))

os.makedirs("../outputs", exist_ok=True)

plt.figure(figsize=(10,5))
plt.plot(all_targets[:300], label="Actual")
plt.plot(all_preds[:300], label="Predicted")
plt.legend()
plt.title("Predicted vs Actual Temperature")
plt.xlabel("Time")
plt.ylabel("Normalized Temperature")
plt.grid(True)

plt.savefig("../outputs/predicted_vs_actual.png")
plt.show()