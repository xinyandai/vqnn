import scipy
from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision.datasets as torch_datasets

_DATASETS_MAIN_PATH = './Datasets'

#_IMAGE_NET_PATH = '/home/xinyan/program/data/ILSVRC2012/'
_IMAGE_NET_PATH = '/data/dataset/ILSVRC2012/'
#_IMAGE_NET_PATH = '/research/jcheng2/xinyan/zzhang/AlexnetandVGG/ILSVRC2012/'

_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'kdd2010': os.path.join(_DATASETS_MAIN_PATH, 'KDD2010'),
    'rcv1': os.path.join(_DATASETS_MAIN_PATH, 'RCV1'),
    'rcv1_binary': os.path.join(_DATASETS_MAIN_PATH, 'RCV1_BINARY'),

    'imagenet': {
        'train': _IMAGE_NET_PATH + 'ILSVRC2012_img_train',
        'val': _IMAGE_NET_PATH + 'ILSVRC2012_img_val',
        'test': _IMAGE_NET_PATH + 'ILSVRC2012_img_test',
    }
}


class RCV1(Dataset):
    def __init__(self, root, train=True, download=True):
        if train:
            data, target = datasets.fetch_rcv1(
                root, subset='test', download_if_missing=download, return_X_y=True)
            scipy.sparse.save()
        else:
            data, target = datasets.fetch_rcv1(
                root, subset='train', download_if_missing=download, return_X_y=True)

        self.data = torch.from_numpy(data.todense())
        self.target = torch.from_numpy(target.todense()).float()
        print("RC1 loaded")

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.shape[0]


class RCV1_BINARY(Dataset):
    def __init__(self, root, train=True, download=True):
        # train_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
        # test_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2'
        if train:
            X, y = datasets.load_svmlight_file(
                os.path.join(root, 'rcv1_test.binary'))
        else:
            X, y = datasets.load_svmlight_file(
                os.path.join(root, 'rcv1_train.binary'))
        print("Loading RCV1_BINARY...")
        y = np.where(y > 0, y, 0)
        self.X = X
        self.y = torch.from_numpy(y).long()

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index].todense()).view(-1), self.y[index]

    def __len__(self):
        return self.y.shape[0]


class KDD2010(Dataset):
    def __init__(self, root, train=True):
        try:
            if train:
                self.X = scipy.sparse.load_npz(
                    os.path.join(root, 'kddb_features.npz'))
                target = np.load(os.path.join(root, 'kddb_labels.npy'))
            else:
                self.X = scipy.sparse.load_npz(
                    os.path.join(root, 'kddb.t_features.npz'))
                target = np.load(os.path.join(root, 'kddb.t_labels.npy'))
            self.y = torch.from_numpy(target).long()

        except:
            print('KDD2010 init failed! root: {}, train: {}'.format(root, train))

    def __getitem__(self, index):
        coo = self.X.getrow(index).tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        sample = torch.sparse.FloatTensor(
            i, v, torch.Size(shape)).to_dense().reshape(-1)
        target = self.y[index]
        return sample, target

    def __len__(self):
        return self.y.shape[0]


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if 'mnist' in name:
        return torch_datasets.MNIST(root=_dataset_path['mnist'],
                                    train=train,
                                    transform=transform, target_transform=target_transform, download=download)
    elif name == 'cifar10':
        return torch_datasets.CIFAR10(root=_dataset_path['cifar10'],
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)
    elif name == 'cifar100':
        return torch_datasets.CIFAR100(root=_dataset_path['cifar100'],
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
    elif name == 'kdd2010':
        return KDD2010(root=_dataset_path['kdd2010'],
                       train=train)

    elif name == 'rcv1':
        return RCV1(root=_dataset_path['rcv1'],
                    train=train, download=download)
    elif name == 'rcv1_binary':
        return RCV1_BINARY(root=_dataset_path['rcv1_binary'], train=train, download=download)
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        return torch_datasets.ImageFolder(root=path,
                                          transform=transform,
                                          target_transform=target_transform)
