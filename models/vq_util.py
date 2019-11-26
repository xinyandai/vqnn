import torch
import numpy as np

code_books = {}
daggers = {}

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def get_code_book(args, dim, ks):
    if (dim, ks) not in code_books:
        location = './codebooks/{}/angular_dim_{}_Ks_{}.fvecs'
        location = location.format('kmeans_codebook', dim, ks)
        codewords = fvecs_read(location)
        book = torch.from_numpy(codewords)

        if args.gpus is not None:
            book = book.cuda()
        book /= torch.norm(book, dim=1, keepdim=True)[0]
        code_books[(dim, ks)] = book

        c_dagger = np.linalg.pinv(codewords.T)
        c_dagger = torch.from_numpy(c_dagger)
        codes_idx = torch.range(1, ks)
        if args.gpus is not None:
            c_dagger = c_dagger.cuda()
            codes_idx = codes_idx.cuda()

        daggers[book] = (c_dagger, codes_idx)
        return book
    else:
        return code_books[(dim, ks)]

def euclid(q, x):
    assert q.shape[1] == x.shape[1]
    x = x.transpose(0, 1)
    q_norm = torch.norm(q, dim=1, keepdim=True)
    x_norm = torch.norm(x, dim=0, keepdim=True)
    return -2.0 * q.mm(x) + q_norm + x_norm


def vq(q, x):
    return torch.argmin(euclid(q, x), dim=1)


def lloyd(X, Ks, n_iter=20):
    centroids = X[np.random.choice(len(X), Ks)]
    codes = vq(X, centroids)
    for _ in range(n_iter):
        for index in range(Ks):
            indices = torch.nonzero(codes == index).squeeze()
            selected = torch.index_select(X, 0, indices)
            centroids[index] = selected.mean(dim=0)
        codes = vq(X, centroids)
    return codes, centroids