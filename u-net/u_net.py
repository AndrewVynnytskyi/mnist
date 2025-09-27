#%%
import torch.nn as nn
import torch

#%%
class UNet(nn.Module):
    def __init__(self, num_classes, down_sample_block, up_sample_clock, conv_block):
        super().__init__()
        self.dwn_lr1 = down_sample_block(3, 64)
        self.dwn_lr2 = down_sample_block(64, 128)
        self.dwn_lr3 = down_sample_block(128, 256)
        self.dwn_lr4 = down_sample_block(256, 512)
        self.dbl_conv = conv_block(512, 1024)

        self.up_lr1 = up_sample_clock(1024, 512)
        self.up_lr2 = up_sample_clock(512, 256)
        self.up_lr3 = up_sample_clock(256, 128)
        self.up_lr4 = up_sample_clock(128, 64)

        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        skip ,out = self.dwn_lr1(x)
        skip1, out = self.dwn_lr2(out)
        skip2, out = self.dwn_lr3(out)
        skip3, out = self.dwn_lr4(out)

        out = self.dbl_conv(out)

        out = self.up_lr1(out, skip3)
        out = self.up_lr2(out, skip2)
        out = self.up_lr3(out, skip1)
        out = self.up_lr4(out, skip)

        return self.conv_1x1(out)