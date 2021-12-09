import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_channel, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride = 2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_channel, kernel_size = 1)

    def forward(self, input): # [C, H, W]
        c1 = self.conv1(input) # [64, H, W]
        p1 = self.pool1(c1) # [64, H/2, W/2]
        c2 = self.conv2(p1) # [128, H/2, W/2]
        p2 = self.pool2(c2) # [128, H/4, W/4]
        c3 = self.conv3(p2) # [256, H/4, W/4]
        p3 = self.pool3(c3) # [256, H/8, W/8]
        c4 = self.conv4(p3) # [512, H/8, W/8]
        p4 = self.pool4(c4) # [512, H/16, W/16]
        c5 = self.conv5(p4) # [1024, H/16, W/16]

        up6 = self.up6(c5)
        merge6 = torch.cat([up6, c4], dim = 1)
        c6 = self.conv6(merge6)
        up7 = self.up7(c6)
        merge7 = torch.cat([up7, c3], dim = 1)
        c7 = self.conv7(merge7)
        up8 = self.up8(c7)
        merge8 = torch.cat([up8, c2], dim = 1)
        c8 = self.conv8(merge8)
        up9 = self.up9(c8)
        merge9 = torch.cat([up9, c1], dim = 1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10

