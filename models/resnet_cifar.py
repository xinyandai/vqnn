'''ResNet-18 Image classfication for cifar-10 with PyTorch

Author 'Sun-qian'.
https://blog.csdn.net/sunqiande88/article/details/80100891
'''
import math
import torch.nn as nn
import torch.nn.functional as F


def init_model(args, model):
    for m in model.modules():
        if isinstance(m, args.conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, args, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            args.conv2d(args, inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            args.conv2d(args, outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                args.conv2d(args, inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, args, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = args.conv2d(args, in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = args.conv2d(args, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = args.conv2d(args, planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                args.conv2d(args, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            args.conv2d(args, 3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(args, block, 64,  num_blocks[0], stride=1)
        self.layer2 = self.make_layer(args, block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(args, block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(args, block, 512, num_blocks[3], stride=2)

        self.fc = args.linear(args, 512 * block.expansion, num_classes)
        init_model(args, self)
        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            51: {'lr': 1e-2},
            71: {'lr': 1e-3, 'weight_decay': 0},
            91: {'lr': 1e-4}
        }

        self.regime_Adam = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            51: {'lr': 1e-3},
            71: {'lr': 5e-4},
            81: {'lr': 1e-4},
            91: {'lr': 1e-5}
        }


    def make_layer(self, args, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(args, self.inchannel, channels, stride))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNetCifar(**kwargs):
    num_classes, depth, dataset, args = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'args'])

    if  dataset == 'cifar10':
        num_classes = num_classes or 10
    elif  dataset == 'cifar100':
        num_classes = num_classes or 100
    else:
        assert False
    if depth == 152:
        return ResNet(args, Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    if depth == 101:
        return ResNet(args, Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    if depth == 50:
        return ResNet(args, Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if depth == 34:
        return ResNet(args, ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
    if depth == 18:
        return ResNet(args, ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
