from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import summary

from utils.ops import *

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features):
        super(ResUnet, self).__init__()
        self.header = Header(in_ch=in_channels, init_features=init_features)
        self.trailer = Trailer(out_ch=out_channels, init_features=init_features)

    def forward(self, x):
        filters = self.header(x)
        ret = self.trailer(x, filters)

        return ret

class Header(nn.Module):
    def __init__(self, in_ch, init_features, depth=5):
        super(Header, self).__init__()
        self.in_ch = in_ch

        self.features = [2**i for i in range(depth+1)] * init_features

        self.header_resample = nn.Sequential(
                (nn.Conv2d(self.in_ch, self.features[0], kernel_size=3, padding=1)),
                (nn.BatchNorm2d(self.features[0])),
                (nn.LeakyReLU(0.1))
        )
        self.header = ResBlock(self.features[0])
        self.down0, self.enc1 = self._init_layer(1, self.features[0], self.features[1], "enc1")
        self.down1, self.enc2 = self._init_layer(2, self.features[1], self.features[2], "enc2")
        self.down2, self.enc3 = self._init_layer(4, self.features[2], self.features[3], "enc3")
        self.down3, self.enc4 = self._init_layer(8, self.features[3], self.features[4], "enc4")
        self.down4, self.bottleneck = self._init_layer(1, self.features[4], self.features[5], "bottleneck")

    def forward(self, x):
        head = self.header(self.header_resample(x))
        enc1 = self.enc1(self.down0(head))
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))
        bottleneck = self.bottleneck(self.down4(enc4))

        return (head, enc1, enc2, enc3, enc4, bottleneck)

    def _init_layer(self, n_block, in_ch, out_ch, name):
        blocks = []
        for i in range(n_block):
            blocks.append((f"{name}_{i}", ResBlock(out_ch)))

        down = Downsample(mode="conv", in_ch=in_ch, out_ch=out_ch)
        enc = nn.Sequential(OrderedDict(blocks))

        return down, enc


class Trailer(nn.Module):
    def __init__(self, out_ch, init_features, depth=5):
        super(Trailer, self).__init__()
        self.out_ch = out_ch
        self.features = [2**i for i in range(depth+1)] * init_features

        self.up5, self.dec4 = self._init_layer(self.features[5], self.features[4], "dec4")
        self.up4, self.dec3 = self._init_layer(self.features[4], self.features[3], "dec3")
        self.up3, self.dec2 = self._init_layer(self.features[3], self.features[2], "dec2")
        self.up2, self.dec1 = self._init_layer(self.features[2], self.features[1], "dec1")
        self.up1, self.trail = self._init_layer(self.features[1], self.features[0], "trail")

        self.output = nn.Conv2d(self.features[0], self.out_ch, kernel_size=1)

    def forward(self, x, _filters):
        (head, enc1, enc2, enc3, enc4, bottleneck) = _filters
        dec4 = torch.cat((enc4, self.up5(bottleneck)), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = torch.cat((enc3, self.up4(dec4)), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = torch.cat((enc2, self.up3(dec3)), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = torch.cat((enc1, self.up2(dec2)), dim=1)
        dec1 = self.dec1(dec1)

        trail = torch.cat((head, self.up1(dec1)), dim=1)
        trail = self.trail(trail)

        output = torch.sigmoid(self.output(trail))
        return output

    def _init_layer(self, in_ch, out_ch,name):
        up = Upsample(mode="deconv", in_ch=in_ch, out_ch=out_ch)
        dec = BasicBlock(in_ch=(out_ch*2), out_ch=out_ch)

        return up, dec


class YoloLayer(nn.Module):
    def __init__(self, anchors, num_classes):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes

        pass



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUnet(in_channels=3, out_channels=4, init_features=32)
    model = model.to(device)

    summary(model, input_size=(3, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 3, 512, 512))

        net = ResUnet(in_channels=3, out_channels=4, init_features=32)
        out = net(img)

        print(out.size())
