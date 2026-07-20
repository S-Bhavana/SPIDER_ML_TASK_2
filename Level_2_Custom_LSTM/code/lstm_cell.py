import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(
            input_size + hidden_size,
            hidden_size
        )

        self.input_gate = nn.Linear(
            input_size + hidden_size,
            hidden_size
        )

        self.output_gate = nn.Linear(
            input_size + hidden_size,
            hidden_size
        )

        self.cell_gate = nn.Linear(
            input_size + hidden_size,
            hidden_size
        )

    def forward(self, x, h_prev, c_prev):

        combined = torch.cat((x, h_prev), dim=1)

        f = torch.sigmoid(self.forget_gate(combined))

        i = torch.sigmoid(self.input_gate(combined))

        o = torch.sigmoid(self.output_gate(combined))

        g = torch.tanh(self.cell_gate(combined))

        c = f * c_prev + i * g

        h = o * torch.tanh(c)

        return h, c


if __name__ == "__main__":

    batch_size = 8

    input_size = 1

    hidden_size = 64

    cell = CustomLSTMCell(input_size, hidden_size)

    x = torch.randn(batch_size, input_size)

    h = torch.zeros(batch_size, hidden_size)

    c = torch.zeros(batch_size, hidden_size)

    h_new, c_new = cell(x, h, c)

    print("Hidden Shape :", h_new.shape)

    print("Cell Shape   :", c_new.shape)