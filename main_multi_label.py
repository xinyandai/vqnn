import torch
import numpy as np
import torch.nn.functional as F
from sklearn import datasets
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import AverageMeter


def topk_precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labels = target.sum().cpu().numpy()

    _, indices = output.float().topk(maxk, 1, True, True)

    res = []
    res.append((torch.round(output) * target).sum().numpy() / labels)
    for k in topk:
        output_k = torch.zeros_like(target, dtype=target.dtype)
        output_k.scatter_(1, indices[:, :k], 1.0)

        prec = (output_k * target).sum().numpy() / len(target)
        res.append(prec)
    return res


class XMLData(Dataset):
    def __init__(self, train=True):
        self.n_classes = 670091
        self.n_features = 135909
        file_name = 'data/Amazon/amazon_{}.txt'.format('train' if train else 'test')
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


def sparse_tensor(data: torch.sparse.FloatTensor):
    shape = data.shape
    data = data.coalesce()
    i = data.indices()
    v = data.values()
    data = torch.sparse.FloatTensor(torch.stack((i[0], i[2])), v, torch.Size((shape[0], shape[2])))
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
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
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
        assert x.shape == target.shape, \
            "pred : {} v.s. labels : {}".format(x.shape, target.shape)
        x = torch.log_softmax(x, dim=0)
        return -torch.sum(x * target)


def forward(model, optimizer, criterion, loader, device, train):
    losses = AverageMeter()
    prec = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(total=len(loader), desc="# Train" if train else "# Test") as p_bar:
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
                optimizer.step()

            p_bar.update(1)
            p_bar.set_description(
                "Loss [%.3f] BCE[%.3f] Top1[%.3f] Top5[%.3f]" % (losses.avg, prec.avg, top1.avg, top5.avg)
            )

    return losses.avg, prec.avg, top1.avg, top5.avg


def main(in_dim, hidden, out_dim, batch_size, lr, epoch, device):
    model = XMLModel(in_dim, hidden, out_dim).to(device)
    criterion = CrossEntropy()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_data = XMLData(False)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False)
    train_data = XMLData(True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False)

    loss, bce, top1, top5 = forward(model, optimizer, criterion, val_loader, device, False)
    print("Pre Train Test \tLoss [%.3f] BCE[%.3f] Top1[%.3f] Top5[%.3f]" % (loss, bce, top1, top5))
    for i in range(epoch):
        loss, bce, top1, top5 = forward(model, optimizer, criterion, train_loader, device, True)
        print("Train Epoch [%d] \tLoss [%.3f] BCE[%.3f] Top1[%.3f] Top5[%.3f]" % (i, loss, bce, top1, top5))
        loss, bce, top1, top5 = forward(model, optimizer, criterion, val_loader, device, False)
        print("Test Epoch [%d] \tLoss [%.3f] BCE[%.3f] Top1[%.3f] Top5[%.3f]" % (i, loss, bce, top1, top5))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main(135909, 128, 670091, 128, 0.0001, 20, device)
