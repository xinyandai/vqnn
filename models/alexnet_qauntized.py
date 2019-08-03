import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnet_quantized']


class AlexNetOWT_BN(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl=3
        self.features = nn.Sequential(
            args.conv2d(args, 3, int(64*self.ratioInfl), kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(64*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            args.conv2d(args, int(64*self.ratioInfl), int(192*self.ratioInfl), kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(192*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            args.conv2d(args, int(192*self.ratioInfl), int(384*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(384*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            args.conv2d(args, int(384*self.ratioInfl), int(256*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            args.conv2d(args, int(256*self.ratioInfl), 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            args.linear(args, 256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            args.linear(args, 4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            args.linear(args, 4096, num_classes),
            nn.BatchNorm1d(1000),
            nn.LogSoftmax()
        )

        self.regime = {
           0: {'optimizer': 'SGD', 'lr': 1e-2,
               'weight_decay': 5e-4, 'momentum': 0.9},
           10: {'lr': 5e-3},
           15: {'lr': 1e-3, 'weight_decay': 0},
           20: {'lr': 5e-4},
           25: {'lr': 1e-4}
        }
        # self.regime = {
        #     0: {'optimizer': 'Adam', 'lr': 5e-3},
        #     20: {'lr': 1e-3},
        #     30: {'lr': 5e-4},
        #     35: {'lr': 1e-4},
        #     40: {'lr': 1e-5}
        # }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_quantized(**kwargs):
    num_classes, dataset, args = map(
        kwargs.get, ['num_classes', 'dataset', 'args'])
    num_classes = num_classes or 1000
    return AlexNetOWT_BN(args, num_classes)
