import torch
import torch.nn as nn
from torchsummary import summary

from utils.ops import *

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features):
        super(Unet, self).__init__()
        features = [1, 2, 4, 8, 16] * init_features

        self.enc1 = BasicBlock(in_channels, features[0], name="enc1")
        self.down1 = Downsample(mode="pool")

        self.enc2 = BasicBlock(features[0], features[1], name="enc2")
        self.down2 = Downsample(mode="pool")

        self.enc3 = BasicBlock(features[1], features[2], name="enc3")
        self.down3 = Downsample(mode="pool")

        self.enc4 = BasicBlock(features[2], features[3], name="enc4")
        self.down4 = Downsample(mode="pool")

        self.bottleneck = BasicBlock(features[3], features[4], name="bottleneck")

        self.upsample4 = Upsample("deconv", features[4], features[3])
        self.dec4 = BasicBlock((features[3]+features[3]), features[3], name="dec4")

        self.upsample3 = Upsample("deconv",features[3], features[2])
        self.dec3 = BasicBlock((features[2]+features[2]), features[2], name="dec3")

        self.upsample2 = Upsample("deconv",features[2], features[1])
        self.dec2 = BasicBlock((features[1]+features[1]), features[1], name="dec2")

        self.upsample1 = Upsample("deconv",features[1], features[0])
        self.dec1 = BasicBlock(features[1], features[0], name="dec1")

        self.conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))

        bottleneck = self.bottleneck(self.down4(enc4))

        dec4 = self.upsample4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upsample3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upsample2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upsample1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        output = torch.sigmoid(self.conv(dec1))
        return output


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
