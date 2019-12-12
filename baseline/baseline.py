import torch
import torch.nn as nn
from torchsummary import summary

from utils.ops import *

class Baseline(nn.Module):
    def __init__(self, in_channels, out_channels, init_features):
        super(Baseline, self).__init__()
        self.header = Header(in_ch=in_channels, init_features=init_features)
        self.trailer = Trailer(out_ch=out_channels,
                                init_features=init_features)

    def forward(self, x):
        enc1, enc2, enc3, enc4, bottleneck = self.header(x)
        ret = self.trailer(enc1, enc2, enc3, enc4, bottleneck)

        return ret

class Header(nn.Module):
    def __init__(self, in_ch, init_features, depth=4):
        super(Header, self).__init__()
        self.in_ch = in_ch

        self.features = [(2**i)*init_features for i in range(depth+1)]

        self.down1, self.enc1 = self._init_layer(self.in_ch, self.features[0], "enc1")
        self.down2, self.enc2 = self._init_layer(self.features[0], self.features[1], "enc2")
        self.down3, self.enc3 = self._init_layer(self.features[1], self.features[2], "enc3")
        self.down4, self.enc4 = self._init_layer(self.features[2], self.features[3], "enc4")
        self.bottleneck = BasicBlock(self.features[3], self.features[4])

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))
        bottleneck = self.bottleneck(self.down4(enc4))

        return enc1, enc2, enc3, enc4, bottleneck

    def _init_layer(self, in_ch, out_ch, name):
        enc = BasicBlock(in_ch, out_ch)
        down = nn.MaxPool2d(kernel_size=2, stride=2)
        return down, enc


class Trailer(nn.Module):
    def __init__(self, out_ch, init_features, depth=4):
        super(Trailer, self).__init__()
        self.out_ch = out_ch
        self.features = [(2**i)*init_features for i in range(depth+1)]
        
        self.up4, self.dec4 = self._init_layer(self.features[4], self.features[3], "dec4")
        self.up3, self.dec3 = self._init_layer(self.features[3], self.features[2], "dec3")
        self.up2, self.dec2 = self._init_layer(self.features[2], self.features[1], "dec2")
        self.up1, self.dec1 = self._init_layer(self.features[1], self.features[0], "dec1")

        self.output = nn.Conv2d(self.features[0], self.out_ch, kernel_size=1)

    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        dec4 = torch.cat((self.up4(bottleneck), enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = torch.cat((self.up3(dec4), enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = torch.cat((self.up2(dec3), enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = torch.cat((self.up1(dec2), enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        output = torch.sigmoid(self.output(dec1))
        
        return output

    def _init_layer(self, in_ch, out_ch,name):
        up = Upsample(mode="deconv", in_ch=in_ch, out_ch=out_ch)
        dec = BasicBlock(in_ch=(out_ch*2), out_ch=out_ch)

        return up, dec


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Baseline(in_channels=1, out_channels=4, init_features=32)
    model = model.to(device)

    summary(model, input_size=(1, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 512, 512))

        net = Baseline(in_channels=1, out_channels=4, init_features=32)
        
        output = net(img)
