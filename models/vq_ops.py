import torch
import torch.nn as nn
from .vq_util import vq
from .vq_util import get_code_book


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