import torch.nn as nn
import torch

# affine_par = True
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

# class ETM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ETM, self).__init__()
#         self.relu = nn.ReLU(True)
#         self.branch0 = BasicConv2d(in_channels, out_channels, 1)
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channels, out_channels, 1),
#             BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
#             BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
#             BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channels, out_channels, 1),
#             BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channels, out_channels, 1),
#             BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
#         self.conv_res = BasicConv2d(in_channels, out_channels, 1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#
#         x = self.relu(x_cat + self.conv_res(x))
#         return x
def resize_tensor(tensor, size):
    """
    使用插值方法将张量调整大小

    Args:
        tensor (torch.Tensor): 输入的张量，形状为 (batch_size, num_channels, height, width)
        size (tuple): 调整后的目标大小，形状为 (new_height, new_width)

    Returns:
        torch.Tensor: 调整大小后的张量，形状为 (batch_size, num_channels, new_height, new_width)
    """
    resized_tensor = nn.functional.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    return resized_tensor

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # x01 x02是低频信号; x1 x2 x3 x4是高频信号
        x01 = x[:, :, 0::2, :] / 2
        # 表示在第三个维度上以步幅为2选择元素，即选择索引为0、2、4、...的元素。
        x02 = x[:, :, 1::2, :] / 2
        # 这意味着只选择索引为1、3、5等奇数位置上的元素。/ 2：这是除法运算符，
        # 将选取的部分 x[:, :, 0::2, :] 的每个元素都除以2，即 / 2 的操作。这样做是为了缩小高频部分的值范围，将其变得更小。
        # 在一些信号处理的方法中，高频部分往往具有较大的值，可能会对某些操作产生较大的影响，例如激活函数的响应范围、优化算法的收敛性等。
        # 通过将高频部分除以2，可以将其数值范围缩小一半，使其相对于低频部分具有更小的权重，从而在一定程度上平衡了高频和低频的影响。
        x1 = x01[:, :, :, 0::2]
        #  是对 x01 在空间维度上进行下采样的操作，步长为2。这样的操作会使得 x01 在宽度维度上减半，
        #  即将每一行的元素进行间隔取样，形状变为 (batch_size, num_channels, height//2, width//2)。
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh
    # ll 子图表示低频部分的信息，lh、hl、hh 分别表示高频部分在不同方向上的信息。

class extract_high_frequency(nn.Module):
    def __init__(self):
        super(extract_high_frequency, self).__init__()
        self.dwt = DWT()
        self.conv_in = nn.Conv2d(768, 256, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        ll, lh, hl, hh = self.dwt(x)
        high_frequency = resize_tensor(hh, (64, 64))
        return high_frequency
