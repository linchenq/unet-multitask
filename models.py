from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import summary

from utils.ops import *

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features, num_anchors, num_classes):
        super(ResUnet, self).__init__()
        self.header = Header(in_ch=in_channels, init_features=init_features)
        self.trailer = Trailer(out_ch=out_channels,
                               init_features=init_features,
                               num_anchors=num_anchors,
                               num_classes=num_classes)

    def forward(self, x):
        head, enc1, enc2, enc3, enc4, bottleneck = self.header(x)
        ret = self.trailer(head, enc1, enc2, enc3, enc4, bottleneck)

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

        return head, enc1, enc2, enc3, enc4, bottleneck

    def _init_layer(self, n_block, in_ch, out_ch, name):
        blocks = []
        for i in range(n_block):
            blocks.append((f"{name}_{i}", ResBlock(out_ch)))

        down = Downsample(mode="conv", in_ch=in_ch, out_ch=out_ch)
        enc = nn.Sequential(OrderedDict(blocks))

        return down, enc


class Trailer(nn.Module):
    def __init__(self, out_ch, init_features, num_anchors, num_classes, depth=5):
        super(Trailer, self).__init__()
        self.out_ch = out_ch
        self.features = [2**i for i in range(depth+1)] * init_features
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.yolo_filter = num_anchors * (5 + num_classes)

        self.up5, self.dec4 = self._init_layer(self.features[5], self.features[4], "dec4")
        self.up4, self.dec3 = self._init_layer(self.features[4], self.features[3], "dec3")
        self.up3, self.dec2 = self._init_layer(self.features[3], self.features[2], "dec2")
        self.up2, self.dec1 = self._init_layer(self.features[2], self.features[1], "dec1")
        self.up1, self.trail = self._init_layer(self.features[1], self.features[0], "trail")

        # yolo layers
        # n_blocks, in_ch, out_ch, num_filters):

        self.yolo3 = YoloBlock(n_blocks=2, in_ch=self.features[5], out_ch=self.features[4], num_filters=self.yolo_filter)
        self.yolo2 = YoloBlock(n_blocks=0, in_ch=self.features[4], out_ch=self.features[3], num_filters=self.yolo_filter)
        self.yolo1 = YoloBlock(n_blocks=0, in_ch=self.features[3], out_ch=self.features[2], num_filters=self.yolo_filter)

        self.output = nn.Conv2d(self.features[0], self.out_ch, kernel_size=1)

    def forward(self, head, enc1, enc2, enc3, enc4, bottleneck):
        yolo3 = self.yolo3(bottleneck)
        dec4 = torch.cat((enc4, self.up5(bottleneck)), dim=1)
        dec4 = self.dec4(dec4)
        yolo2 = self.yolo2(dec4)

        dec3 = torch.cat((enc3, self.up4(dec4)), dim=1)
        dec3 = self.dec3(dec3)
        yolo1 = self.yolo1(dec3)

        dec2 = torch.cat((enc2, self.up3(dec3)), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = torch.cat((enc1, self.up2(dec2)), dim=1)
        dec1 = self.dec1(dec1)

        trail = torch.cat((head, self.up1(dec1)), dim=1)
        trail = self.trail(trail)

        output = torch.sigmoid(self.output(trail))
        
        return yolo3, yolo2, yolo1, output

    def _init_layer(self, in_ch, out_ch,name):
        up = Upsample(mode="deconv", in_ch=in_ch, out_ch=out_ch)
        dec = BasicBlock(in_ch=(out_ch*2), out_ch=out_ch)

        return up, dec


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResUnet(in_channels=3, out_channels=4, init_features=32, num_anchors=3, num_classes=6)
    model = model.to(device)

    summary(model, input_size=(3, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 3, 512, 512))

        net = ResUnet(in_channels=3, out_channels=4, init_features=32, num_anchors=3, num_classes=6)
        # output, yolo3, yolo2, yolo1 = net(img)
        # print(f"yolo layers are {yolo3.size()}, {yolo2.size()}, {yolo1.size()}")
        # print(f"output layer is {output.size()}")
        
        output = net(img)
        print(f"yolo layers are {output[0].size()}, {output[1].size()}, {output[2].size()}")
        print(f"output layer is {output[3].size()}")
        