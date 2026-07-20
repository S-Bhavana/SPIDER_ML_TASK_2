import torch
import torch.nn as nn
from lstm_cell import CustomLSTMCell


class CustomLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, output_size=12):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        seq_len = x.size(1)

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):

            h, c = self.lstm_cell(x[:, t, :], h, c)

        out = self.fc(h)

        return out.unsqueeze(-1)


if __name__ == "__main__":

    model = CustomLSTM()

    x = torch.randn(16, 72, 1)

    y = model(x)

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)