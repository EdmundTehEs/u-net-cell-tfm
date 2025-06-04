# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:18:57 2025

@author: mbiww
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, sf=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, sf)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(sf, sf*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(sf*2, sf*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(sf*4, sf*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(sf*8, sf*16)
        self.up4 = nn.ConvTranspose2d(sf*16, sf*8, 2, 2)
        self.dec4 = ConvBlock(sf*16, sf*8)
        self.up3 = nn.ConvTranspose2d(sf*8, sf*4, 2, 2)
        self.dec3 = ConvBlock(sf*8, sf*4)
        self.up2 = nn.ConvTranspose2d(sf*4, sf*2, 2, 2)
        self.dec2 = ConvBlock(sf*4, sf*2)
        self.up1 = nn.ConvTranspose2d(sf*2, sf, 2, 2)
        self.dec1 = ConvBlock(sf*2, sf)
        self.final = nn.Conv2d(sf, out_ch, 1)

    def forward(self, x):
        x = x / 255.0
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        c5 = self.bottleneck(p4)
        u6 = self.up4(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.dec4(u6)
        u7 = self.up3(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.dec3(u7)
        u8 = self.up2(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.dec2(u8)
        u9 = self.up1(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.dec1(u9)
        out = self.final(c9)
        return torch.sigmoid(out)