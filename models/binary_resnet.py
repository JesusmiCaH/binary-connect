from __future__ import absolute_import

'''
ResNet for CIFAR-10/100 Dataset (Only BasicBlock is used).
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition (CVPR 2016). https://arxiv.org/abs/1512.03385

The Wide ResNet model is the same as ResNet except for the number of channels 
is double/quadruple/k-time larger in every basicblock.
Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. Sergey Zagoruyko and Nikos Komodakis
Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146

P.S. 
Following the previous repository "https://github.com/HobbitLong/RepDistiller", the num_filters of the first conv is doubled in ResNet-8x4/32x4.
The wide ResNet model in the "/RepDistiller/models/wrn.py" is almost the same as ResNet model in "/RepDistiller/models/resnet.py".
For example, wrn_40_2 in "/RepDistiller/models/wrn.py" almost equals to resnet38x2 in "/RepDistiller/models/resnet.py". 
The only difference is that resnet38x2 has additional three BN layers, which leads to 2*(16+32+64)*k parameters [k=2 in this comparison].
Therefore, it is recommanded to directly use this file for the implementation of the Wide ResNet model.
'''
from torch.autograd import Function
import torch.nn as nn
from torch.nn.common_types import _size_2_t

__all__ = ['resnet']
class BinaryActivation(Function):
    # @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

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
        b_weight = self.weight.sign()
        b_bias = self.bias.sign() if self.bias != None else self.bias

        return nn.functional.conv2d(input, b_weight, b_bias, stride=self.stride, padding=self.padding, groups=self.groups)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Binary_conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Binary_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = BinaryActivation.apply(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = BinaryActivation.apply(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = BinaryActivation.apply(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = BinaryActivation.apply(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = BinaryActivation.apply(out)
        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = Binary_conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.fc)
        return feat_m
        
    def forward(self, x, is_feat=False):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = BinaryActivation.apply(x)    # 32x32
        f0 = x

        x = self.layer1(x)  # 32x32
        f1 = x
        x = self.layer2(x)  # 16x16
        f2 = x
        x = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            return [f0, f1, f2, f3, f4], x
        else:
            return x

def binary_resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)

# wrn_40_1 (We use the wrn notation to be consistent with the previous work)
def binary_resnet38(**kwargs):
    return ResNet(38, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet116(**kwargs):
    return ResNet(116, [16, 16, 32, 64], 'basicblock', **kwargs)

def binary_resnet200(**kwargs):
    return ResNet(200, [16, 16, 32, 64], 'basicblock', **kwargs)

# wrn_16_2 (We use the wrn notation to be consistent with the previous work)
def binary_resnet14x2(**kwargs):
    return ResNet(14, [16, 32, 64, 128], 'basicblock', **kwargs)

# wrn_16_4 (We use the wrn notation to be consistent with the previous work)
def binary_resnet14x4(**kwargs):
    return ResNet(14, [32, 64, 128, 256], 'basicblock', **kwargs)

# wrn_40_2 (We use the wrn notation to be consistent with the previous work)
def binary_resnet38x2(**kwargs):
    return ResNet(38, [16, 32, 64, 128], 'basicblock', **kwargs)

def binary_resnet110x2(**kwargs):
    return ResNet(110, [16, 32, 64, 128], 'basicblock', **kwargs)

def binary_resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)

def binary_resnet20x4(**kwargs):
    return ResNet(20, [32, 64, 128, 256], 'basicblock', **kwargs)

def binary_resnet26x4(**kwargs):
    return ResNet(26, [32, 64, 128, 256], 'basicblock', **kwargs)

def binary_resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)

# wrn_40_4 (We use the wrn notation to be consistent with the previous work)
def binary_resnet38x4(**kwargs):
    return ResNet(38, [32, 64, 128, 256], 'basicblock', **kwargs)

def binary_resnet44x4(**kwargs):
    return ResNet(44, [32, 64, 128, 256], 'basicblock', **kwargs)

def binary_resnet56x4(**kwargs):
    return ResNet(56, [32, 64, 128, 256], 'basicblock', **kwargs)

def binary_resnet110x4(**kwargs):
    return ResNet(110, [32, 64, 128, 256], 'basicblock', **kwargs)

if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = binary_resnet32(num_classes=100)
    feats, logit = net(x, is_feat=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    
    #for i, m in enumerate(net.get_feat_modules()):
    #    print(i, m)
    
    num_params_stu = (sum(p.numel() for p in net.parameters())/1000000.0)
    print('Total params_stu: {:.3f} M'.format(num_params_stu))