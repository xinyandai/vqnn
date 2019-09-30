import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


code_books = {}


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
        book /= torch.max(torch.abs(book), dim=1, keepdim=True)[0]
        code_books[(dim, ks)] = book
        return book
    else:
        return code_books[(dim, ks)]


def euclid(q, x):
    x = x.reshape((x.shape[0], -1))
    q = q.reshape((q.shape[0], -1))
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


class _VQLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, book=None, idx=None):
        # what is idx?
        ctx.save_for_backward(input, weight)
        if book is None:
            output = input.mm(weight.t())
        else:
            ks, dim = book.shape
            N, D = input.shape
            H, _ = weight.shape
            M = D // dim
            dist = input.view(N * M, dim).mm(book.t()).view(N, M, ks)
            selected = torch.gather(dist, dim=2,
                                    index=idx.view(H, M).t().expand((N, M, H)))
            output = selected.sum(dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight, None, None


class _VQConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(cxt, input, weight, bias=None,
                stride=1, padding=0, dilation=1, groups=1):
        cxt.save_for_backward(input, weight, bias)
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias = cxt.saved_variables
        grad_input = grad_weight = grad_bias = None
        if cxt.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output)
        if cxt.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output)
        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias


class VQLinear(nn.Linear):
    def __init__(self, args, in_features, out_features, bias=True):
        super(VQLinear, self).__init__(
            in_features, out_features, bias)
        self.codes = None
        self.args = args
        self.dim = args.dim
        self.ks = args.ks
        self.register_buffer("code_book",
                             get_code_book(args, self.dim, self.ks))

    def quantize(self, tensor_):
        tensor = tensor_.clone()
        x = tensor.view(-1, self.dim)
        codes = vq(x, self.code_book)
        torch.index_select(
            self.code_book, 0, codes, out=x[:, :])
        return tensor, codes

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data, self.codes = self.quantize(self.weight.org)
        # self.weight.data, self.codes = self.quantize(self.weight.data)

        out = nn.functional.linear(input, self.weight)
        # out = _VQLinearFunction.apply(
        #     input, self.weight, self.quantize.code_book, self.codes)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class VQConv2d(nn.Conv2d):

    def __init__(self, args, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(VQConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.codes = None
        self.args = args
        self.dim = 3 if in_channels == 3 else args.dim
        self.dim = args.dim
        self.ks = args.ks
        self.register_buffer("code_book",
                             get_code_book(args, self.dim, self.ks))

    def quantize(self, tensor_):
        # input is in shape of (N, C_in, H, W)
        tensor = tensor_.permute(0, 2, 3, 1).contiguous()
        x = tensor.view(-1, self.dim)
        codes = vq(x, self.code_book)
        torch.index_select(
            self.code_book, 0, codes, out=x[:, :])
        return tensor.permute(0, 3, 1, 2), codes

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data, self.codes = self.quantize(self.weight.org)
        # self.weight.data, self.codes = self.quantize(self.weight.data)

        out = nn.functional.conv2d(
            input, self.weight, None, self.stride,
            self.padding, self.dilation, self.groups)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class BCLinear(nn.Linear):
    def __init__(self, args, in_features, out_features, bias=True):
        super(BCLinear, self).__init__(
            in_features, out_features, bias)

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = self.weight.org.sign()
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BCConv2d(nn.Conv2d):

    def __init__(self, args, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BCConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = self.weight.org.sign()

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class BNNLinear(BCLinear):
    def __init__(self, args, in_features, out_features, bias=True):
        super(BNNLinear, self).__init__(
            args, in_features, out_features, bias)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = input.data.sign()
        return super(BNNLinear, self).forward(input)


class BNNConv2d(BCConv2d):

    def __init__(self, args, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BNNConv2d, self).__init__(
            args, in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = input.data.sign()
        return super(BNNConv2d, self).forward(input)


class Linear(nn.Linear):
    def __init__(self, args, in_features, out_features, bias=True):
        super(Linear, self).__init__(
            in_features, out_features, bias)


class Conv2d(nn.Conv2d):

    def __init__(self, args, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
