import torch.nn as nn

__all__ = ['cnn', 'fcn']

class CNN(nn.Module):
    def __init__(self, args, H=1024):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            args.conv2d(args, 1, 5, 5, 1),
            args.activation(args, inplace=True),
            nn.MaxPool2d(2, 2),

            args.conv2d(args, 5, 50, 5, 1),
            args.activation(args, inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            args.dropout(args) ,
            args.linear(args, 4*4*50, H),
            args.activation(args, inplace=True),
            args.dropout(args) ,
            args.linear(args, H, 10)
        )

        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            10: {'lr': 1e-2},
            20: {'lr': 1e-3, 'weight_decay': 0},
            30: {'lr': 1e-4},
            40: {'lr': 1e-5},
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4*4*50)
        x = self.classifier(x)
        return x

class FCN(nn.Module):
    def __init__(self, args, D_in=784, H=1024, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FCN, self).__init__()
        self.linear1 = args.linear(args, D_in, H)
        self.linear2 = args.linear(args, H, num_classes)
        self.relu = args.activation(args, inplace=True)
        self.dropout = args.dropout(args)

        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            10: {'lr': 1e-2},
            20: {'lr': 1e-3, 'weight_decay': 0},
            30: {'lr': 1e-4},
            40: {'lr': 1e-5},
        }

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 28 * 28)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y_pred = self.linear2(x)
        return y_pred

def cnn(**kwargs):
    args = kwargs['args']
    return CNN(args)

def fcn(**kwargs):
    args = kwargs['args']
    return FCN(args)