from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class _Conv2d3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Conv2d3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.leaky1 = nn.LeakyReLU(0.1)
        # self.leaky1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky1(out)

        return out


class _Conv2d1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Conv2d1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.leaky1 = nn.LeakyReLU(0.1)
        # self.leaky1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky1(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.block1 = _Conv2d3x3(in_ch, out_ch)
        self.block2 = _Conv2d3x3(out_ch, out_ch)

    def forward(self, x):
       out = self.block1(x)
       out = self.block2(out)

       return out


class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResBlock, self).__init__()
        self.block1 = _Conv2d3x3(in_ch, in_ch)
        self.block2 = _Conv2d3x3(in_ch, in_ch)

    def forward(self, x):
        res = x
        out = self.block1(x)
        out = self.block2(out)
        out += res

        return out


class YoloBlock(nn.Module):
    def __init__(self, n_blocks, in_ch, out_ch, num_filters):
        super(YoloBlock, self).__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append((f"YOLO_{i}_1x1_{in_ch}_{out_ch}", _Conv2d1x1(in_ch, out_ch)))
            blocks.append((f"YOLO_{i}_3x3_{in_ch}_{out_ch}", _Conv2d3x3(out_ch, in_ch)))
        blocks.append((f"YOLO_final_1x1_{in_ch}_{out_ch}", _Conv2d1x1(in_ch, out_ch)))
        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.conv3x3 = _Conv2d3x3(out_ch, in_ch)
        self.conv1x1 = _Conv2d1x1(in_ch, num_filters)

    def forward(self, x):
        out = self.blocks(x)
        out = self.conv3x3(out)
        out = self.conv1x1(out)

        return out


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
        out = self.block(x)
        return out


class Downsample(nn.Module):
    def __init__(self, mode, in_ch=-1, out_ch=-1):
        super(Downsample, self).__init__()
        if mode == "pool":
            self.block = nn.MaxPool2d(kernel_size=2, stride=2)
        elif mode == "conv":
            self.block = nn.Sequential(
                                (nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)),
                                (nn.BatchNorm2d(out_ch)),
                                (nn.ReLU(inplace=True))
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.block(x)
        return out