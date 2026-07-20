import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class WeatherDataset(Dataset):

    def __init__(self, csv_path, input_hours=72, output_hours=12, train=True):

        # Read CSV
        df = pd.read_csv(csv_path)

        # Temperature column
        temperature = df["T (degC)"].values.reshape(-1, 1)

        # Downsample by averaging every 6 rows
        usable = (len(temperature) // 6) * 6
        temperature = temperature[:usable]

        temperature = temperature.reshape(-1, 6, 1).mean(axis=1)

        # Normalize
        self.scaler = MinMaxScaler()
        temperature = self.scaler.fit_transform(temperature)

        X = []
        Y = []

        for i in range(len(temperature) - input_hours - output_hours):

            X.append(temperature[i:i + input_hours])

            Y.append(
                temperature[
                    i + input_hours:
                    i + input_hours + output_hours
                ]
            )

        X = np.array(X)
        Y = np.array(Y)

        split = int(0.8 * len(X))

        if train:
            self.X = X[:split]
            self.Y = Y[:split]
        else:
            self.X = X[split:]
            self.Y = Y[split:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        return (
            torch.tensor(self.X[index], dtype=torch.float32),
            torch.tensor(self.Y[index], dtype=torch.float32)
        )


def get_dataloaders(csv_path, batch_size=64):

    train_dataset = WeatherDataset(csv_path, train=True)

    test_dataset = WeatherDataset(csv_path, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = get_dataloaders(
        "../data/jena_climate_2009_2016.csv"
    )

    x, y = next(iter(train_loader))

    print("Input Shape :", x.shape)

    print("Target Shape:", y.shape)