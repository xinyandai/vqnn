import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['alexnet']


class AlexNetOWT_BN(nn.Module):

    def __init__(self, args, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(
            args.conv2d(args, 3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            args.conv2d(args, 64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            args.conv2d(args, 192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            args.conv2d(args, 384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            args.conv2d(args, 256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            args.linear(args, 256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            args.linear(args, 4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            args.linear(args, 4096, num_classes)
        )

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

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

def alexnet(**kwargs):
    num_classes, dataset, args = map(
        kwargs.get, ['num_classes', 'dataset', 'args'])
    num_classes = num_classes or 1000
    return AlexNetOWT_BN(args, num_classes)
