import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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


def neq(q, x):
    return d_neq(q, x)


def d_neq(q, x):
    assert q.shape[1] == x.shape[1]
    ip = q.mm(x.transpose(0, 1))
    codes = torch.argmax(ip, dim=1)
    u = ip.gather(dim=1, index=codes.view(-1, 1))
    return codes, u


def p_neq(q, x, cuda=True):
    assert q.shape[1] == x.shape[1]

    c_dagger, codes_idx = daggers[x]
    # calculate probability, complexity: O(d*K)
    ip = q.mm(c_dagger.transpose(0, 1))
    l1_norms = torch.norm(ip, p=1, dim=1, keepdim=True)
    probability = torch.abs(ip) / l1_norms

    # choose codeword with probability
    r = torch.rand(probability.size()[0])
    if cuda:
        r = r.cuda()
    rs = r.view(-1, 1).expand_as(probability)

    comp = torch.cumsum(probability, dim=1) >= rs - (1e-5)
    comp = comp.type(torch.float32)
    comp /= codes_idx
    codes = torch.argmax(comp, dim=1)

    selected_p = ip.gather(dim=1, index=codes.view(-1, 1))
    u = torch.mul(torch.sign(selected_p), l1_norms)

    return codes, u


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
        # quantize weight nad decompress it to self.weight
        self.weight.data, self.codes = self.quantize(self.weight.org)
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
        self.ks = args.ks
        self.register_buffer("code_book",
                             get_code_book(args, self.dim, self.ks))

    def quantize(self, tensor_):
        tensor = tensor_.permute(0, 2, 3, 1).contiguous()
        x = tensor.view(-1, self.dim)
        codes = vq(x, self.code_book)
        torch.index_select(
            self.code_book, 0, codes, out=x[:, :])
        return tensor.permute(0, 3, 1, 2), codes

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        # quantize weight, decompress it to self.weight
        self.weight.data, self.codes = self.quantize(self.weight.org)
        out = nn.functional.conv2d(
            input, self.weight, None, self.stride,
            self.padding, self.dilation, self.groups)

        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out



class _VQActivationLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_, dim, code_book, depth):
        tensor = tensor_.clone()
        x = tensor.view(-1, dim)
        compressed = torch.empty_like(x)
        sum = torch.zeros_like(x)
        for _ in range(depth):
            codes, u = neq(x, code_book)
            torch.index_select(
                code_book, 0, codes, out=compressed[:, :])
            compressed *= u
            sum += compressed
            x -= compressed
        return sum.reshape_as(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class _VQActivationConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_, dim, code_book, depth):
        # batch size, channels, height, width
        # batch size, height, width, channels
        tensor = tensor_.permute(0, 2, 3, 1).contiguous()
        x = tensor.view(-1, dim)
        compressed = torch.empty_like(x)
        sum = torch.zeros_like(x)
        for _ in range(depth):
            codes, u = neq(x, code_book)
            torch.index_select(
                code_book, 0, codes, out=compressed[:, :])
            compressed *= u
            sum += compressed
            x -= compressed
        return sum.reshape_as(tensor).permute(0, 3, 1, 2)

    @staticmethod
    def backward(cxt, grad_output):
        return grad_output, None, None, None


class VQActivation(nn.Module):
    def __init__(self, args, inplace=True):
        super(VQActivation, self).__init__()
        self.inplace = inplace
        self.codes = None
        self.args = args
        self.dim = args.dim
        self.ks = args.ks
        self.register_buffer("code_book",
                             get_code_book(args, self.dim, self.ks))


    def forward(self, x):
        if len(x.size()) == 4:
            qx  =  _VQActivationConv2d.apply(x, self.dim, self.code_book, self.args.r)
        else:
            qx = _VQActivationLinear.apply(x, self.dim, self.code_book, self.args.r)
        if self.inplace:
            x.data = qx
            return x
        else:
            return qx

sum_complexity = 0.0
sum_accelerate = 0.0
sum_memory = 0.0
sum_lookup = 0.0
threshold = 2.0
id = 0

class VQActivationLinear(VQLinear):
    def __init__(self, args, in_features, out_features, bias=True):
        super(VQActivationLinear, self).__init__(
            args, in_features, out_features, bias)
        self.speed_up = 2.0 * in_features * out_features / (
                in_features * self.ks * args.r if self.ks == self.dim else 1.0 +
                2.0 * out_features * (in_features / self.dim) * args.r
        )
        self.memory = out_features * in_features
        self.lookup = out_features * (in_features / self.dim) * self.ks
        self.vqa = self.speed_up > threshold and in_features not in [784, 32 * 32, 224 * 224]

        global id
        self.id = id
        id += 1

        if not self.vqa:
            self.speed_up = 1.0

        self.complexity = None


    def forward(self, x):
        global sum_accelerate
        global sum_complexity
        global sum_memory
        global sum_lookup

        self.af = open('activations/{}_{}_{}_linear_I{}_O{}.txt'.format(
            self.id, self.args.model, self.args.dataset, self.in_features, self.out_features), 'ab')
        np.savetxt(self.af, x.detach().cpu().numpy())
        self.af.close()

        if self.complexity is None:
            self.complexity = self.in_features * self.out_features
            sum_complexity += self.complexity
            sum_accelerate += self.complexity / self.speed_up
            sum_lookup += self.lookup
            sum_memory += self.memory
            logging.info("linear  I: {}\tO: {}\tsize: {}\t\n"
                         "speed up: {}\t complexity: {}\taccelerate: {}\t\n"
                         "overall speed up : {}\tsum_complexity :{}\tsum accelerate: {}\n"
                         "memory blow up: {}\t sum memory: {}\t sum look up table: {}\n\n"
                .format(
                self.in_features, self.out_features, x.size(1),
                self.speed_up, self.complexity, self.complexity / self.speed_up,
                sum_complexity / sum_accelerate, sum_complexity, sum_accelerate,
                sum_lookup / sum_memory, sum_memory, sum_lookup))

        # if self.vqa:
        #    x = _VQActivationLinear.apply(x, self.dim, self.code_book, self.args.r)
        out = nn.functional.linear(
            x, self.weight, bias=self.bias)
        return out


class VQActivationConv2d(VQConv2d):
    def __init__(self, args, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(VQActivationConv2d, self).__init__(
            args, in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.speed_up = 2.0 * kernel_size * kernel_size * out_channels * in_channels / (
                in_channels * self.ks * (args.r if self.ks == self.dim else 1.0) +
                2.0 * kernel_size * kernel_size * out_channels * (in_channels / self.dim) * args.r
        )
        self.memory = kernel_size**2 * out_channels * in_channels
        self.lookup = kernel_size**2 * out_channels * (in_channels / self.dim) * self.ks
        self.vqa = self.speed_up > threshold and not self.in_channels == 3


        global id
        self.id = id
        id += 1

        if not self.vqa:
            self.speed_up = 1.0
        self.complexity = None

    def forward(self, x):
        global sum_accelerate
        global sum_complexity
        global sum_memory
        global sum_lookup

        self.af = open(
            'activations/{}_{}_{}_conv_I{}_O{}_W{}_H{}.txt'.format(
            self.id, self.args.model, self.args.dataset,
            self.in_channels, self.out_channels,
            x.size(2), x.size(3)), 'ab')
        np.savetxt(self.af,
                   x.permute(0, 2, 3, 1)
                   .contiguous()
                   .view(-1, self.in_channels)
                   .detach().cpu().numpy())
        self.af.close()

        if self.complexity is None:
            self.complexity =  self.in_channels * self.out_channels \
                               * self.kernel_size[0] * self.kernel_size[1] \
                               * x.size(2) * x.size(3)
            sum_complexity += self.complexity
            sum_accelerate += self.complexity / self.speed_up
            sum_lookup += self.lookup
            sum_memory += self.memory
            logging.info("kernel: {}\tI: {}\tO: {}\tsize: {}\t\n"
                         "speed up: {}\t complexity: {}\taccelerate: {}\t\n"
                         "overall speed up : {}\tsum_complexity :{}\tsum accelerate: {}\n"
                         "memory blow up: {}\t sum memory: {}\t sum look up table: {}\n\n".format(
                self.kernel_size, self.in_channels, self.out_channels, x.size(2),
                self.speed_up, self.complexity, self.complexity / self.speed_up,
                sum_complexity / sum_accelerate, sum_complexity, sum_accelerate,
                sum_lookup / sum_memory, sum_memory, sum_lookup))
        # if self.vqa :
        #     x = _VQActivationConv2d.apply(x, self.dim, self.code_book, self.args.r)
        out = nn.functional.conv2d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
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
        if input.size(1) not in [784, 32 * 32, 224 * 224]:
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


class Identical(nn.Module):
    def __init__(self, args):
        super(Identical, self).__init__()

    def forward(self, x):
        return x