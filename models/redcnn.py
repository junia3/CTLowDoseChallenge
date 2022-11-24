import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=5, padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class depthwise_separable_trconv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_trconv, self).__init__()
        self.depthwise = nn.ConvTranspose2d(nin, nin * kernels_per_layer, kernel_size=5, padding=0, groups=nin)
        self.pointwise = nn.ConvTranspose2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class RED_CNN(nn.Module):
    def __init__(self):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=0)


    def forward(self, x):
        # encoder
        residual_1 = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        residual_2 = out
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        residual_3 = out
        out = F.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(F.relu(out))
        out = self.tconv3(F.relu(out))
        out += residual_2
        out = self.tconv4(F.relu(out))
        out = self.tconv5(F.relu(out))
        out += residual_1
        out = torch.clamp(out, 0, 1)
        return out

class RED_CNN_LITE(nn.Module):
    def __init__(self):
        super(RED_CNN_LITE, self).__init__()
        self.conv1 = depthwise_separable_conv(1, 96, 96)
        self.conv2 = depthwise_separable_conv(96, 1, 96)
        self.conv3 = depthwise_separable_conv(96, 1, 96)
        self.conv4 = depthwise_separable_conv(96, 1, 96)
        self.conv5 = depthwise_separable_conv(96, 1, 96)
        self.tconv1 = depthwise_separable_trconv(96, 1, 96)
        self.tconv2 = depthwise_separable_trconv(96, 1, 96)
        self.tconv3 = depthwise_separable_trconv(96, 1, 96)
        self.tconv4 = depthwise_separable_trconv(96, 1, 96)
        self.tconv5 = depthwise_separable_trconv(96, 1, 1)


    def forward(self, x):
        # encoder
        residual_1 = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        residual_2 = out
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        residual_3 = out
        out = F.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(F.relu(out))
        out = self.tconv3(F.relu(out))
        out += residual_2
        out = self.tconv4(F.relu(out))
        out = self.tconv5(F.relu(out))
        out += residual_1
        out = torch.clamp(out, 0, 1)
        return out