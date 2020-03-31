import copy
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from RecurrentUNet.models.SRU import SRU
from modelsummary import summary


class ConvBlock_b(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock_b,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=k_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class UpBlock_b(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpBlock_b,self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.tconv2d = nn.ConvTranspose2d(ch_in, ch_in, 4, 2, 1, bias=False)
        self.o_conv = ConvBlock_b(ch_in+ch_out, ch_out, k_size=3, stride=1, padding=1, bias=False)

    def forward(self, in1, in2):
        in1 = self.tconv2d(in1)
        x = torch.cat((in1, in2), dim=1)
        x = self.o_conv(x)
        return x


class Backbone(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Backbone, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.ConvBlock_b4 = ConvBlock_b(ch_in, ch_in*2)
        self.ConvBlock_cc = ConvBlock_b(ch_in*2, ch_in*2)
        self.UpBlock_b4 = UpBlock_b(ch_in*2, ch_out)

    def forward(self, x):
        e4 = self.ConvBlock_b4(x)
        cc = self.Maxpool(e4)
        cc = self.ConvBlock_cc(cc)
        d4 = self.UpBlock_b4(cc, e4)
        return d4


class R_UNet(nn.Module):
    def __init__(self):
        super(R_UNet, self).__init__()
        self.prev_t = None
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.ConvBlock_b1 = ConvBlock_b(6, 8)
        self.ConvBlock_b2 = ConvBlock_b(8, 16)
        self.ConvBlock_b3 = ConvBlock_b(16, 32)
        self.SRU = SRU(32, 64, 3, backnet=Backbone(32, 64))
        self.UpBlock_b3 = UpBlock_b(32, 32)
        self.UpBlock_b2 = UpBlock_b(32, 16)
        self.UpBlock_b1 = UpBlock_b(16, 8)
        self.ConvBlock_f = ConvBlock_b(8, 3)

    def forward(self, x):
        if self.prev_t == None:
            self.prev_t = torch.zeros(x.size()).cuda(1)
        x = torch.cat((x, self.prev_t), dim=1)
        e1 = self.ConvBlock_b1(x)
        e2 = self.Maxpool(e1)
        e2 = self.ConvBlock_b2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.ConvBlock_b3(e3)
        e4 = self.Maxpool(e3)
        e4 = e4.unsqueeze(0)
        d4, _ = self.SRU(e4)
        d3 = self.UpBlock_b3(d4[0].squeeze(1), e3)
        d2 = self.UpBlock_b2(d3, e2)
        d1 = self.UpBlock_b1(d2, e1)
        x = self.ConvBlock_f(d1)
        self.prev_t = x

        return x
        

if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = R_UNet().cuda()
    summary(model, torch.zeros((1, 3, 512, 512)).cuda(), show_input=True)
    summary(model, torch.zeros((1, 3, 512, 512)).cuda(), show_input=False)

    y = model(torch.zeros((1, 3, 512, 512)).cuda())
    