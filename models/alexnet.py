import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, args, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            args.conv2d(args, 3, 64, kernel_size=11, stride=4, padding=2),
            args.activation(args, inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            args.conv2d(args, 64, 192, kernel_size=5, padding=2),
            args.activation(args, inplace=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            args.conv2d(args, 192, 384, kernel_size=3, padding=1),
            args.activation(args, inplace=True),
            args.conv2d(args, 384, 256, kernel_size=3, padding=1),
            args.activation(args, inplace=True),
            args.conv2d(args, 256, 256, kernel_size=3, padding=1),
            # args.activation(args, inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            args.dropout(args) ,
            args.linear(args, 256 * 6 * 6, 4096),
            args.activation(args, inplace=True),
            args.dropout(args) ,
            args.linear(args, 4096, 4096),
            args.activation(args, inplace=True),
            args.linear(args, 4096, num_classes),
        )

        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }

        self.regime_Adam = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    num_classes, dataset, args = map(
        kwargs.get, ['num_classes', 'dataset', 'args'])
    num_classes = num_classes or 1000
    model = AlexNet(args, num_classes)
    if args.pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=True)
        model.load_state_dict(state_dict)
    
    return model
