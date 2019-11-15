from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import summary

from utils.ops import *

class ResUnet(nn.Module):

    def __init__(self, in_channels, out_channels, init_features):
        super(ResUnet, self).__init__()
        pass


class Header(nn.Module):
    def __init__(self, in_ch, out_ch, init_features, depth=4):
        super(Header, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.features = [2**i for i in range(depth+1)] * init_features

        self.header = nn.Sequential(OrderedDict([
                            (nn.Conv2d(in_ch, self.features[0], kernel_size=3, padding=1)),
                            (nn.BatchNorm2d(in_ch)),
                            (nn.LeakyReLU(0.1))
        ]))

        self.enc1, self.down1 = self._init_layer(1, self.features[0], self.features[0], "enc1")
        self.enc2, self.down2 = self._init_layer(1, self.features[0], self.features[1], "enc2")
        self.enc3, self.down3 = self._init_layer(1, self.features[1], self.features[2], "enc3")
        self.enc4, self.down4 = self._init_layer(1, self.features[2], self.features[3], "enc4")

        self.bottleneck = ResBlock(features[3], features[4])

    def forward(self, x):
        head = self.header(x)
        enc1 = self.enc1(head)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))
        bottleneck = self.bottleneck(self.down4(enc4))

        return enc1, enc2, enc3, enc4, bottleneck

    def _init_layer(self, n_block, in_ch, out_ch, name):
        blocks = []
        for i in range(n_block):
            blocks.append((f"{name}_{i}", ResBlock(in_ch, out_ch)))
            in_ch = out_ch

        enc = nn.Sequential(OrderedDict(blocks))
        down = Downsample(mode="pool", in_ch=out_ch)
        return enc, down

class YoloLayer(nn.Module):
    def __init__(self):
        super(YoloLayer, self).__init__()
        pass
    # TODO :  Add yolo parts

    #     self.upsample4 = Upsample("deconv", features[4], features[3])
    #     self.dec4 = BasicBlock((features[3]+features[3]), features[3], name="dec4")

    #     self.upsample3 = Upsample("deconv",features[3], features[2])
    #     self.dec3 = BasicBlock((features[2]+features[2]), features[2], name="dec3")

    #     self.upsample2 = Upsample("deconv",features[2], features[1])
    #     self.dec2 = BasicBlock((features[1]+features[1]), features[1], name="dec2")

    #     self.upsample1 = Upsample("deconv",features[1], features[0])
    #     self.dec1 = BasicBlock(features[1], features[0], name="dec1")

    #     self.conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # def forward(self, x):
    #     enc1 = self.enc1(x)
    #     enc2 = self.enc2(self.down1(enc1))
    #     enc3 = self.enc3(self.down2(enc2))
    #     enc4 = self.enc4(self.down3(enc3))

    #     bottleneck = self.bottleneck(self.down4(enc4))

    #     dec4 = self.upsample4(bottleneck)
    #     dec4 = torch.cat((dec4, enc4), dim=1)
    #     dec4 = self.dec4(dec4)

    #     dec3 = self.upsample3(dec4)
    #     dec3 = torch.cat((dec3, enc3), dim=1)
    #     dec3 = self.dec3(dec3)

    #     dec2 = self.upsample2(dec3)
    #     dec2 = torch.cat((dec2, enc2), dim=1)
    #     dec2 = self.dec2(dec2)

    #     dec1 = self.upsample1(dec2)
    #     dec1 = torch.cat((dec1, enc1), dim=1)
    #     dec1 = self.dec1(dec1)

    #     output = torch.sigmoid(self.conv(dec1))
    #     return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(in_channels=3, out_channels=4, init_features=32)
    model = model.to(device)

    summary(model, input_size=(3, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 3, 512, 512))

        net = Unet(in_channels=3, out_channels=4, init_features=32)
        out = net(img)

        print(out.size())
