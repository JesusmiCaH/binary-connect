import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

class BinaryActivation(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input-1e-6).sign()

    # @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > 1] = 0
        grad_input[input < -1] = 0
        return grad_input
    
class Binary_conv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    def forward(self, input):
        self.weight.clip(-1,1)
        self.bias.clip(-1,1)
        # save the continuous weight here and binarize them
        # the continuous weight will be returned back after backward
        self.saved_weight = self.weight.data
        self.saved_bias = self.bias.data
        # w_rand = torch.rand(self.weight.shape)*2 - 1   #binarize strategy: treat weight as the prob of being 1.
        # b_rand = torch.rand(self.bias.shape)*2 - 1   
        # w_rand, b_rand = w_rand.cuda(), b_rand.cuda()

        # self.weight.data = (self.weight.data - w_rand).sign()
        # self.bias.data = (self.bias.data - b_rand).sign()
        self.weight.data = self.weight.data.sign()
        self.bias.data = self.bias.data.sign()
        return nn.functional.conv2d(input, self.weight, self.bias, padding=self.padding, groups=self.groups)


class Binary_linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
    def forward(self, input):
        
        self.weight.clip(-1,1)
        self.bias.clip(-1,1)

        self.saved_weight = self.weight.data
        self.saved_bias = self.bias.data

        # w_rand = torch.rand(self.weight.shape)*2 - 1   #binarize strategy: treat weight as the prob of being 1.
        # b_rand = torch.rand(self.bias.shape)*2 - 1   
        # w_rand, b_rand = w_rand.cuda(), b_rand.cuda()

        # self.weight.data = (self.weight.data - w_rand).sign()
        # self.bias.data = (self.bias.data - b_rand).sign()
        self.weight.data = self.weight.data.sign()
        self.bias.data = self.bias.data.sign()
        return nn.functional.linear(input, self.weight, self.bias)

class Shiftbase_Batchnorm2d(nn.Module):
    def __init__(self, feature_num, dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(feature_num))
        self.epsilon = 1e-4
        self.coefficient = nn.Parameter(torch.rand(feature_num))
        self.cal_dims = [x for x in range(dim) if x != 1]

    def forward(self, x):
        self.mean = x.mean(self.cal_dims)
        for dim in self.cal_dims:
            self.mean = self.mean.unsqueeze(dim)

        x = x - self.mean
        self.var = x.var(self.cal_dims)
        
        coefficient = self.weight / torch.sqrt(self.var + self.epsilon)
        if not self.training:
            coefficient = coefficient.sign()
            # coefficient = coefficient.sign() * torch.pow(2, torch.round(torch.log2(coefficient.abs())) )
            self.coefficient.data = coefficient
        # Upgrade dim so that multiplication can go through
        for dim in self.cal_dims:
            coefficient = coefficient.unsqueeze(dim)

        return x * coefficient


class Custom_Padding(nn.Module):
    def __init__(self, padding = [0,1,0,1]) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(x, self.padding, mode="constant", value = 0)

