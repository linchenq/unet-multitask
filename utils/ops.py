from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.leaky1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(mid_ch, in_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.leaky2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky2(out)

        out += res

        return res


class Upsample(nn.Module):
    def __init__(self, mode, in_ch=-1, out_ch=-1):
        super(Upsample, self).__init__()
        if mode == "nearest":
            self.block = F.interpolate(scale_factor=2, mode="nearest")
        elif mode == "deconv":
            self.block = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, mode, in_ch=-1):
        super(Downsample, self).__init__()
        if mode == "pool":
            self.block = nn.MaxPool2d(kernel_size=2, stride=2)
        elif mode == "conv":
            self.block = nn.Sequential(OrderedDict([
                                (nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)),
                                (nn.BatchNorm2d(in_ch)),
                                (nn.ReLU(inplace=True))
            ]))
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.block(x)