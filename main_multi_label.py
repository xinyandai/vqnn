import torch
import numpy as np
import torch.nn.functional as F
from sklearn import datasets
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import AverageMeter
from models.vq_ops import VQLinear, Linear
from collections import namedtuple


def topk_precision(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    :param output: (N. K)
    :param target: (N, K)
    :param topk:
    :return:
    """
    maxk = max(topk)
    output = output.cpu()
    target = target.cpu()
    labels = target.sum().cpu().numpy()

    _, indices = output.float().topk(maxk, 1, True, True)

    res = []
    res.append(((output > -0.30).float() * target).sum().numpy() / labels)
    for k in topk:
        output_k = torch.zeros_like(target, dtype=target.dtype)
        output_k.scatter_(1, indices[:, :k], 1.0)
        prec = ((output_k * target).sum(dim=1) > 0)\
                   .float().sum().numpy() / len(target)
        res.append(prec)
    return res


class XMLData(Dataset):
    def __init__(self, n, d, file_name):
        self.n_classes = n
        self.n_features = d 
        self.X, self.y = datasets.load_svmlight_file(
            file_name, multilabel=True, offset=1, n_features=self.n_features)
        self._get_y = self._get_y_dense

    def _get_x(self, index):
        data = self.X.getrow(index).tocoo()
        i = torch.LongTensor(np.vstack((data.row, data.col)))
        v = torch.FloatTensor(data.data)
        data = torch.sparse.FloatTensor(i, v, torch.Size(data.shape))
        return data

    def _get_y_dense(self, index):
        classes = self.y[index]
        y = torch.zeros(self.n_classes)
        y[torch.LongTensor(classes)] = 1.0
        return y / len(classes)

    def _get_y_sparse(self, index):
        cols = self.y[index]
        zeros = [0 for _ in cols]
        ones = [1 for _ in cols]
        i = torch.LongTensor(np.vstack((zeros, cols)))
        v = torch.FloatTensor(ones)
        data = torch.sparse.FloatTensor(i, v, torch.Size((1, self.n_classes)))
        return data

    def __getitem__(self, index):
        return self._get_x(index), self._get_y(index)

    def __len__(self):
        return len(self.y)


class AmazonData(XMLData):
    def __init__(self, train):
        file_name = 'data/Amazon/amazon_{}.txt'.format('train' if train else 'test')
        super(AmazonData, self).__init__(670091, 135909, file_name)


class Wiki10Data(XMLData):
    def __init__(self, train):
        file_name = 'data/wiki10/wiki10_{}.txt'.format('train' if train else 'test')
        super(Wiki10Data, self).__init__(30938, 101938, file_name)


def sparse_tensor(data: torch.sparse.FloatTensor):
    shape = data.shape
    data = data.coalesce()
    i = data.indices()
    v = data.values()
    data = torch.sparse.FloatTensor(
        torch.stack((i[0], i[2])), v, torch.Size((shape[0], shape[2])))
    return data


class SparseLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        output = torch.spmm(x, self.weight.t())
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class XMLModel(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super(XMLModel, self).__init__()
        self.fc1 = SparseLinear(in_dim, hidden)
        # self.fc2 = torch.nn.Linear(hidden, out_dim)
        Args = namedtuple('Args', 'ks dim gpus')
        args = Args(ks=256, dim=2, gpus="0")
        self.fc2 = VQLinear(args, hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=0)
        return x


class CrossEntropy(torch.nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x, target):
        """
        :param x: N * C
        :param target: N * C
        :return:
        """
        return -torch.sum(x * target)


def forward(model, optimizer, criterion, loader, device, train):
    losses = AverageMeter()
    prec = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if train:
        model.train()
    else:
        model.eval()

    with tqdm(total=len(loader), ) as p_bar:
        for inputs, target in loader:
            model.train()
            inputs = sparse_tensor(inputs)
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)

            prec_bce, prec1, prec5 = topk_precision(output.data, target, topk=(1, 5))
            prec.update(prec_bce.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            losses.update(loss.item(), inputs.size(0))

            if train:
                loss.backward()
                for p in list(model.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
                optimizer.step()

            p_bar.update(1)
            p_bar.set_description(
                "%s Loss [%.3f] BCE[%.3f] Top1[%.3f] Top5[%.3f]" %
                ("# Train" if train else "# Test",
                 losses.avg, prec.avg, top1.avg, top5.avg)
            )
    return losses.avg, prec.avg, top1.avg, top5.avg


def main(train_data, val_data, hidden, batch_size, lr, epoch, device):
    model = XMLModel(train_data.n_features, hidden, train_data.n_classes).to(device)
    criterion = CrossEntropy()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False)

    forward(model, optimizer, criterion, val_loader, device, train=False)
    for i in range(epoch):
        forward(model, optimizer, criterion, train_loader, device, train=True)
        forward(model, optimizer, criterion, val_loader, device, train=False)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    MyXMLData = Wiki10Data
    val_data = MyXMLData(train=False)
    train_data = MyXMLData(train=True)

    main(train_data, val_data,
         hidden=128, batch_size=128,
         lr=0.0001, epoch=20, device=device)
