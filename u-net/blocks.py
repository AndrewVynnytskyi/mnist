import torch.nn as nn
import torch

#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, padding=1),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = identity + out
        out = self.relu(out)

        return out

#%%
class ResDownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.mxp = nn.MaxPool2d(2, 2)

    def forward(self, x):
        down = self.res1(x)
        p = self.mxp(down)

        return down, p

#%%

class ResUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.ConvTranspose2d(in_channels, in_channels//2, stride=2, kernel_size=2)
        self.res1 = ResidualBlock(in_channels, out_channels)


    def forward(self, x1, x2):
        up = self.conv_up(x1)
        out = torch.cat([up, x2], 1)
        return self.res1(out)

#%%
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#%%

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dbl_conv = DoubleConvolution(in_channels, out_channels)
        self.dwn_samp = nn.MaxPool2d(2, 2)

    def forward(self, x):
        down = self.dbl_conv(x)
        p = self.dwn_samp(down)

        return down, p

#%%

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_samp = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,
                                          stride=2)
        self.dbl_conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1, x2):
        out = self.up_samp(x1)
        out = torch.cat([out, x2], 1)
        return self.dbl_conv(out)
