import torch
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
