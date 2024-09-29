import torch.nn as nn
import torch.nn.functional as F

def _upsample_like(src, tar):
    # 将 src 移动到与 tar 相同的设备 (GPU)
    # src = src.to(device)
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src
class PUAModule(nn.Module):
    def __init__(self, channel):
        super(PUAModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, dilation=1, groups=channel//2)
        self.conv2 = nn.Conv2d(channel, 2*channel, kernel_size=3, stride=1, padding=1, dilation=2, groups=channel//2)
        self.conv3 = nn.Conv2d(2*channel, channel, kernel_size=3, stride=2, padding=1, dilation=3, groups=channel//2)

        # self.conv4 = nn.Conv2d(channel, channel/2, kernel_size=1, stride=1, padding=1)
        # self.classifier = nn.Conv2d(channel, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(2*channel)
        self.bn3 = nn.BatchNorm2d(channel)
        # self.bn4 = nn.BatchNorm2d(channel)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        res_x = _upsample_like(res_x, x)
        x = x + res_x
        res_x_2 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        # res_x_2 = _upsample_like(res_x_2, x)
        # x = x + res_x_2
        # res_x_3 = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        res_x_2 = _upsample_like(res_x_2, x)
        x = x + res_x_2
        # x = self.conv4(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.leaky_relu(x)
        # x = self.classifier(x)
        return x