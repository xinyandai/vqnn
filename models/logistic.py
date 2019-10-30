import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, args, input_dim, output_dim, bce_loss=False):
        super(LogisticRegression, self).__init__()
        self.fc = args.linear(args, input_dim, output_dim)
        self.regime_SGD = {
            0: {'optimizer': 'SGD', 'lr': 1e-3},
        }
        self.regime_Adam = {
            0: {'optimizer': 'Adam', 'lr': 1e-3},
        }
        if bce_loss:
            self.criterion = torch.nn.BCEWithLogitsLoss

    def forward(self, x):
        outputs = self.fc(x)
        return outputs


def logistic(**kwargs):
    bce_loss = False
    num_classes, dataset, args = map(
        kwargs.get, ['num_classes', 'dataset', 'args'])
    if dataset == 'mnist_linear':
        num_classes = num_classes or 10
        input_dim = 784
    elif dataset == 'kdd2010':
        num_classes = num_classes or 2
        input_dim = 29890095
    elif dataset == 'rcv1':
        num_classes = num_classes or 103
        input_dim = 47236
        bce_loss = True
    elif dataset == 'rcv1_binary':
        num_classes = num_classes or 2
        input_dim = 47236

    return LogisticRegression(args, input_dim, num_classes, bce_loss)
