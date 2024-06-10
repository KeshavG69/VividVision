import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )  # 256x256
        self.down2 = nn.Sequential(
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
        )  # 128x128
        self.down3 = nn.Sequential(
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
        )  # 64x64
        self.down4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )  # 32x32
        self.down5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )  # 16x16
        self.down6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )  # 8x8
        self.down7 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )  # 4x4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )  # 2x2
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )  # 4x4
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512 * 2,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Dropout(0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 8x8
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512 * 2,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Dropout(0.5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 16x16
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512 * 2,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )  # 32x32
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512 * 2,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )  # 64x64
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256 * 2,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # 128x128
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128 * 2, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 256x256
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64 * 2, out_channels=2, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )  # 512x512, 2 channels for colorization

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))

        return self.final_up(torch.cat([up7, d1], 1))
