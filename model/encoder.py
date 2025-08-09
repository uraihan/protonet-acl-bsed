import torch.nn as nn
import torch
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(0.1)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training,
                        inplace=True)  # TODO: Find out if we need this
        return out


# TODO: Fix this (?)
class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, avg_pool=True, dropblock_size=5):
        super(ResNet, self).__init__()

        self.in_channels = 1
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 64, stride=2, drop_block=True,
                                       block_size=dropblock_size)

        # TODO: Eliminate these magic numbers
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((4, int(2048/4*64)))

        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((4, int(2048/4*64)))

    def _make_layer(self, block, out_channels, stride=1, drop_block=False,
                    block_size=1, features=None):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(
            self.in_channels, out_channels, stride, downsample
        ))

        self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        num_samples, seq_len, mel_bins = x.shape
        x = x.view(-1, 1, seq_len, mel_bins)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_avg_pool(x)

        # TODO: decide which one is the most appropriate out of these two return
        # return x.view(x.size(0), -1)
        return x
