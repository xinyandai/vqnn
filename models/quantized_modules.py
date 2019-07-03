import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

from .pyvq import vq

D = 8
K = 256
CODE_BOOK = torch.randn(size=(K, D)).cuda()
CODE_BOOK /= torch.norm(CODE_BOOK, dim=1, keepdim=True)


def identical_quantize(tensor_):
    return tensor_.clone()

def sign_quantize(tensor_):
    return tensor_.sign()


def vector_quantize(tensor_):
    tensor = tensor_.clone()
    flat = tensor.flatten()
    size = flat.shape[0]
    rows = size // D
    compressed = flat[:rows * D].view(rows, D)

    codes = vq(compressed, CODE_BOOK)
    torch.index_select(CODE_BOOK, 0, codes, out=compressed[:, :])
    return tensor

quantize = vector_quantize
print("using quantizer: ", quantize)

class QuantizedLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(QuantizedLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if quantize is sign_quantize and input.size(1) != 784:
            input.data=quantize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=quantize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class QuantizedConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(QuantizedConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if quantize is sign_quantize and input.size(1) != 3:
            input.data = quantize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=quantize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
