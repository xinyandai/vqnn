import torch.nn as nn
import torchvision.transforms as transforms
import math

__all__ = ['resnet']


def conv3x3(args, in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return args.conv2d(args, in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=1, bias=False)


def init_model(args, model):
    for m in model.modules():
        if isinstance(m, args.conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(args, inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(args, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, args, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = args.conv2d(
            args, inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = args.conv2d(args, planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = args.conv2d(
            args, planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, args, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                args.conv2d(args, self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(args, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, args, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = args.conv2d(args, 3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(args, block, 64, layers[0])
        self.layer2 = self._make_layer(args, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(args, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = args.linear(args, 512 * block.expansion, num_classes)

        init_model(args, self)
        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }
        self.regime_Adam = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            30: {'lr': 1e-3},
            60: {'lr': 5e-4},
            70: {'lr': 1e-4},
            80: {'lr': 1e-5}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, args, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inflate = 5
        self.inplanes = 16 * self.inflate
        n = int((depth - 2) / 6)
        self.conv1 = args.conv2d(args, 3, 16 * self.inflate,
                                 kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * self.inflate)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(args, block, 16 * self.inflate, n)
        self.layer2 = self._make_layer(
            args, block, 32 * self.inflate, n, stride=2)
        self.layer3 = self._make_layer(
            args, block, 64 * self.inflate, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = args.linear(args, 64 * self.inflate, num_classes)

        init_model(args, self)
        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            81: {'lr': 1e-2},
            122: {'lr': 1e-3, 'weight_decay': 0},
            164: {'lr': 1e-4}
        }

        self.regime_Adam = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            101: {'lr': 1e-3},
            142: {'lr': 5e-4},
            184: {'lr': 1e-4},
            220: {'lr': 1e-5}
        }


def resnet(**kwargs):
    num_classes, depth, dataset, args = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'args'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(args=args,  num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(args=args, num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(args=args, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(args=args, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(args=args, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])
    #
    # elif dataset == 'cifar10':
    #     num_classes = num_classes or 10
    #     depth = depth or 18
    #     return ResNet_cifar10(args=args, num_classes=num_classes,
    #                           block=BasicBlock, depth=depth)
    #
    # elif dataset == 'cifar100':
    #     num_classes = num_classes or 100
    #     depth = depth or 18
    #     return ResNet_cifar10(args=args, num_classes=num_classes,
    #                           block=BasicBlock, depth=depth)

    elif dataset in ["cifar10", "cifar100"]:
        from .resnet_cifar import ResNetCifar
        return ResNetCifar(**kwargs)
