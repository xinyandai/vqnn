import logging
import torch
import torch.nn as nn
import numpy as np
from .vq_util import vq


def pq(q, c, M, d):
    codes = [vq(q[:, m * d : (m + 1) * d], c[m, :, :])
             for m in range(M)]
    return codes


def decompress(c, compressed, codes, M, d):
    for m in range(M):
        torch.index_select(
            c[m, :, :], 0, codes[m], out=compressed[:, m * d : (m + 1) * d])


class _VQCodebookConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_, code_book, depth, M, dim):
        # batch size, channels, height, width
        # batch size, height, width, channels
        tensor = tensor_.permute(0, 2, 3, 1).contiguous()
        x = tensor.view(-1, tensor_.size(1))
        compressed = torch.empty_like(x)
        sum = torch.zeros_like(x)

        for i in range(depth):
            codes = pq(x, code_book[i, :, :, :], M, dim)
            decompress(code_book[i, :, :, :], compressed, codes, M, dim)
            sum += compressed
            x -= compressed

        return sum.reshape_as(tensor).permute(0, 3, 1, 2)

    @staticmethod
    def backward(cxt, grad_output):
        return grad_output, None, None, None, None



sum_complexity = 0.0
sum_accelerate = 0.0
sum_memory = 0.0
sum_lookup = 0.0
threshold = 0.0
id = 0


class VQCodebookConv2d(nn.Conv2d):
    def __init__(self, args, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(VQCodebookConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

        self.codes = None
        self.args = args
        self.dim = 3 if in_channels == 3 else args.dim
        self.ks = args.ks
        self.r = args.r
        self.M = in_channels // self.dim
        self.register_buffer("code_book", torch.empty((args.r, self.M, self.ks, self.dim), dtype=torch.float32))

        self.speed_up = 2.0 * kernel_size * kernel_size * out_channels * in_channels / (
                in_channels * self.ks * (self.r if self.ks == self.dim else 1.0) +
                2.0 * kernel_size * kernel_size * out_channels * (in_channels / self.dim) * self.r
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
        self.af_codebook = None

    def forward(self, x):
        global sum_accelerate
        global sum_complexity
        global sum_memory
        global sum_lookup

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

        if self.args.save_activation:
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
        elif self.vqa:
            if self.af_codebook is None:
                self.af_codebook = 'activations/{}_{}_{}_conv_I{}_O{}_W{}_H{}.txt' \
                                   '.M{}_Ks{}_r{}_dim{}_rq_{}_{}_codebook'.format(
                                    self.id, self.args.model, self.args.dataset,
                                    self.in_channels, self.out_channels,
                                    x.size(2), x.size(3), self.M, self.ks,
                                    self.args.r, self.dim, self.args.r, self.ks)
                self.code_book.view(-1)[:] = torch.from_numpy(np.fromfile(self.af_codebook, dtype=np.float32))
            x = _VQCodebookConv2d.apply(x, self.code_book, self.r, self.M, self.dim)

        out = nn.functional.conv2d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)
        return out
