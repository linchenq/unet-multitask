from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, name):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
                            (name + "conv1", nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)),
                            (name + "bn1", nn.BatchNorm2d(out_ch)),
                            (name + "relu1", nn.ReLU(inplace=True)),
                            (name + "conv2", nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)),
                            (name + "bn2", nn.BatchNorm2d(out_ch)),
                            (name + "relu2", nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)


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