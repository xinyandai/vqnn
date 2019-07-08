import torch.nn as nn
from .pyvq import creat_quantize, sign_quantize


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizedLinear, self).__init__(
            in_features, out_features, bias)
        self.quantize = creat_quantize(self)

    def forward(self, input):

        if self.quantize is sign_quantize and input.size(1) != 784:
            input.data = self.quantize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data =  self.quantize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizedConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantizedConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.quantize = creat_quantize(self)


    def forward(self, input):
        if  self.quantize is sign_quantize and input.size(1) != 3:
            input.data =  self.quantize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data= self.quantize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
