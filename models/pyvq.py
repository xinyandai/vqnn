import torch
import torch.nn as nn
import numpy as np


def euclid(q: torch.Tensor, x: torch.Tensor):
    x = x.reshape((x.shape[0], -1))
    q = q.reshape((q.shape[0], -1))
    x = x.transpose(0, 1)
    q_norm = torch.norm(q, dim=1, keepdim=True)
    x_norm = torch.norm(x, dim=0, keepdim=True)
    return -2.0 * q.mm(x) + q_norm + x_norm


def vq(q: torch.Tensor, x: torch.Tensor):
    return torch.argmin(euclid(q, x), dim=1)


def lloyd(X: torch.Tensor, Ks: int, n_iter: int = 20):
    centroids = X[np.random.choice(len(X), Ks)]
    codes = vq(X, centroids)
    for _ in range(n_iter):
        for index in range(Ks):
            indices = torch.nonzero(codes == index).squeeze()
            selected = torch.index_select(X, 0, indices)
            centroids[index] = selected.mean(dim=0)
        codes = vq(X, centroids)
    return codes, centroids


def identical_quantize(tensor_: torch.Tensor):
    return tensor_.clone()


def sign_quantize(tensor_: torch.Tensor):
    return tensor_.sign()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


code_books = {}

config = DotDict()
config.quantize = "vq"
config.dim = 4
config.ks = 16


def get_code_book(dim: int, ks: int):
    if (dim, ks) not in code_books:
        book = torch.randn(size=(ks, dim)).cuda()
        book /= torch.max(torch.abs(book), dim=1, keepdim=True)[0]
        code_books[(dim, ks)] = book
        return book
    else:
        return code_books[(dim, ks)]


def creat_quantize(module: nn.Module):
    if config.quantize in ["BNN", "BC"]:
        return sign_quantize
    elif config.quantize == "identical":
        return identical_quantize
    elif config.quantize == "vq":
        if isinstance(module, nn.Linear):
            return LinearQuantize(dim=config.dim, ks=config.ks)
        elif isinstance(module, nn.Conv2d):
            assert module.transposed is False
            dim = 3 if module.in_channels == 3 else config.dim
            return ConvQuantize(dim=dim, ks=config.ks)

    assert False, "No matched {} quantize for {}" \
        .format(config.quantize, module)


class LinearQuantize(object):
    def __init__(self, dim: int, ks: int):
        self.dim = dim
        self.code_book = get_code_book(dim, ks)

    def compress(self, tensor_: torch.Tensor):
        w = tensor_.view(-1, self.dim)
        return vq(w, self.code_book)

    def __call__(self, tensor_: torch.Tensor):
        tensor = tensor_.clone()
        x = tensor.view(-1, self.dim)
        codes = vq(x, self.code_book)
        torch.index_select(
            self.code_book, 0, codes, out=x[:, :])
        return tensor


class ConvQuantize(object):
    def __init__(self, dim: int, ks: int):
        self.dim = dim
        self.code_book = get_code_book(dim, ks)

    def compress(self, tensor_: torch.Tensor):
        tensor = tensor_.permute(0, 2, 3, 1).contiguous()
        return vq(tensor.view(-1, self.dim), self.code_book)

    def __call__(self, tensor_: torch.Tensor):
        tensor = tensor_.permute(0, 2, 3, 1).contiguous()
        x = tensor.view(-1, self.dim)
        codes = vq(x, self.code_book)
        torch.index_select(
            self.code_book, 0, codes, out=x[:, :])
        return tensor.permute(0, 3, 1, 2)
