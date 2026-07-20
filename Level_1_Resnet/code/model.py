import torch
import torch.nn as nn

from residual_block import ResidualBlock


class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Initial Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual Layers
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            ResidualBlock(32, 32)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


if __name__ == "__main__":

    model = ResNet()

    x = torch.randn(4, 3, 32, 32)

    y = model(x)

    print(model)

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)