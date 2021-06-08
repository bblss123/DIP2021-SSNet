import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections

import random

class SSNet(nn.Module):

    def __init__(self, batch_norm=False, load_weights=False):
        super(SSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size = 4, stride = 2, padding = 1)
        self.conv2_bn = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, kernel_size = 4, stride = 2, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1) 
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1) # 256 * 6 * 8
        self.conv7_bn = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.conv8_bn = nn.BatchNorm2d(512)
        

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(16)
        self.deconv6 = nn.ConvTranspose2d(16, 8, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(8)
        self.deconv7 = nn.ConvTranspose2d(8, 4, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(4)
        self.deconv8 = nn.ConvTranspose2d(4, 1, 4, 2, 1)
        # self.deconv8_bn = nn.BatchNorm2d(1)

        # if not load_weights:
        #     mod = torchvision.models.vgg16(pretrained=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.conv7_bn(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.conv8_bn(self.conv8(x)), 0.2)

        # print(x.shape)

        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = F.relu(self.deconv7_bn(self.deconv7(x)))
        # x = F.relu(torch.tanh(self.deconv5(x)))
        # x = F.relu(torch.tanh(self.deconv5(x)))
        x = F.relu(torch.tanh(self.deconv8(x)))
        # x = self.deconv5_bn(x)

        return x