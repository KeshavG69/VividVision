import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                512, 1, stride=1, kernel_size=4, padding=1, padding_mode="reflect"
            )
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
