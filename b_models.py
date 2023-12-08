import torch
import torch.nn as nn
from b_layers import BinaryActivation, Binary_conv2d, Binary_linear, Shiftbase_Batchnorm2d, Custom_Padding

class MatthieuConnectNet(nn.Module):
    def __init__(self, num_classes = 10, activate = 'ReLU'):
        super(MatthieuConnectNet, self).__init__()

        self.conv1_1 = Binary_conv2d(3, 128, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(128)
        if activate == 'ReLU':
            self.activate = nn.ReLU()
        elif activate == 'BiLU':
            self.activate = BinaryActivation.apply
        self.conv1_2 = Binary_conv2d(128, 128, 3, 1, 1)
        self.pooling1 = nn.MaxPool2d(2)
        self.bn1_2 = nn.BatchNorm2d(128)
        # relu here again

        self.conv2_1 = Binary_conv2d(128, 256, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(256)
        # relu here again
        self.conv2_2 = Binary_conv2d(256, 256, 3, 1, 1)
        self.pooling2 = nn.MaxPool2d(2)
        self.bn2_2 = nn.BatchNorm2d(256)
        # relu here again

        self.conv3_1 = Binary_conv2d(256, 512, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(512)
        # relu here again
        self.conv3_2 = Binary_conv2d(512, 512, 3, 1, 1)
        self.pooling3 = nn.MaxPool2d(2)
        self.bn3_2 = nn.BatchNorm2d(512)
        # relu here again

        self.fc1 = Binary_linear(512*4*4, 1024)
        self.bn4_1 = nn.BatchNorm1d(1024)
        # relu here again
        self.fc2 = Binary_linear(1024, 1024)
        self.bn4_2 = nn.BatchNorm1d(1024)
        # relu here again
        self.fc3 = Binary_linear(1024, num_classes)
    def forward(self, x):
        x = self.bn1_1(self.conv1_1(x))
        x = self.activate(x)
        x = self.bn1_2(self.pooling1(self.conv1_2(x)))
        x = self.activate(x)

        x = self.bn2_1(self.conv2_1(x))
        x = self.activate(x)
        x = self.bn2_2(self.pooling2(self.conv2_2(x)))
        x = self.activate(x)

        x = self.bn3_1(self.conv3_1(x))
        x = self.activate(x)
        x = self.bn3_2(self.pooling3(self.conv3_2(x)))
        x = self.activate(x)

        x = x.view(x.shape[0], -1)
        x = self.activate(self.bn4_1(self.fc1(x)))
        x = self.activate(self.bn4_2(self.fc2(x)))
        out = self.fc3(x)
        return out


class BinaryConnectNet(nn.Module):
    def __init__(self, num_classes = 10, layers = [64,64,256,256], activate = 'ReLU', needs_feat = True):
        super(BinaryConnectNet, self).__init__()
        self.needs_feat = needs_feat
        self.conv1 = Binary_conv2d(3, layers[0], 3, 1, 1)
        self.pooling1 = nn.MaxPool2d(2)
        self.bn1 = Shiftbase_Batchnorm2d(layers[0], 4)
        if activate == 'ReLU':
            self.activate = nn.ReLU()
        elif activate == 'BiLU':
            self.activate = BinaryActivation.apply
            
        # self.conv2 = Binary_conv2d(layers[0], layers[1], 3, 1, 1)
        self.conv2_1 = Binary_conv2d(layers[0], layers[0], 3, 1, 1, groups=layers[0])
        self.conv2_2 = Binary_conv2d(layers[0], layers[1], 1)
        self.pooling2 = nn.Identity()
        self.bn2 = Shiftbase_Batchnorm2d(layers[1], 4)
        # relu here again

        # self.conv3 = Binary_conv2d(layers[1], layers[2], 3, 1, 1)
        self.conv3_1 = Binary_conv2d(layers[1], layers[1], 3, 1, 1, groups=layers[1])
        self.conv3_2 = Binary_conv2d(layers[1], layers[2], 1)
        self.pooling3 = nn.Identity()
        self.bn3 = Shiftbase_Batchnorm2d(layers[2], 4)
        # relu here again

        # self.conv4 = Binary_conv2d(layers[2], layers[3], 3, 1, 1)
        self.conv4_1 = Binary_conv2d(layers[2], layers[2], 3, 1, 1, groups=layers[2])
        self.conv4_2 = Binary_conv2d(layers[2], layers[3], 1)
        self.bn4 = Shiftbase_Batchnorm2d(layers[3], 4)
        # relu here again

        self.fc1 = Binary_linear(layers[3]*16*16, num_classes)
        # self.bn4 = Shiftbase_Batchnorm2d(1024, 2)
        # # relu here again
        # self.fc2 = Binary_linear(1024, num_classes)
    def forward(self, x):
        x = self.pooling1(self.conv1(x))
        # x = self.pooling1(self.conv1_2(self.conv1_1(x)))
        x = x.clip(-128,127)
        x = self.activate(self.bn1(x))
        # 经过激活后的x值均为+-1
        
        # x = self.pooling2(self.conv2(x))
        x = x + self.conv2_1(x)
        x = self.pooling2(self.conv2_2(x))
        x = x.clip(-128,127)
        # 但在这个卷积层，因为每个特征值是由一百多层的卷积核累加起来的，所以这里x的值会变成一百多，但是大小不重要，重要的是符号。
        # 所以卷积结果必须存进areg里，如果要更加保真，我们需要给这层特征值加一个上限，比如说20。
        x = self.activate(self.bn2(x))
        # print('2Layer----------------------------------------------------')
        # print(x)

        # x = self.pooling3(self.conv3(x))
        x = x + self.conv3_1(x)
        x = self.pooling3(self.conv3_2(x))
        x = x.clip(-128,127)
        x = self.activate(self.bn3(x))
        # print('3Layer----------------------------------------------------')
        # print(x.shape)

        # x = self.conv4(x)
        x = x + self.conv4_1(x)
        x = self.conv4_2(x)
        x = x.clip(-128,127)

        x = self.bn4(x)
        x = self.activate(x)
        f1 = x
        # print('4Layer----------------------------------------------------')
        # print(x.shape)
        f2 = x

        x = x.view(x.shape[0], -1)
        # x = self.activate(self.bn4(self.fc1(x)))
        # out = self.fc2(x)
        out = self.fc1(x)
        if not self.needs_feat:
            return out
        else:
            return [f1, f2], out


class LiBNet(nn.Module):
    def __init__(self, num_classes = 10, layers = [64,256], group_dependency = 1):
        super(LiBNet, self).__init__()
        # basic unit of a layer:
        # binarize--convolution--relu--batchnormalization
        self.bin = BinaryActivation.apply
        self.padder = Custom_Padding([1,2,1,2])
        self.conv1 = Binary_conv2d(3, layers[0], 4, 1, 0)
        self.bn1 = Shiftbase_Batchnorm2d(layers[0], 4)

        self.pooling1 = nn.MaxPool2d(2)
        
        '''Depthwise_conv_1'''
        # depthwise conv
        # bin here
        # pad here
        self.conv2_1 = Binary_conv2d(layers[0], layers[0], 4, 1, 0, groups=layers[0]//(2**group_dependency))
        # self.conv2_1 = nn.Conv2d(layers[0], layers[0], 4, 1, 0, groups=layers[0]//(2**group_dependency))
        # relu here
        self.bn2_1 = Shiftbase_Batchnorm2d(layers[0], 4)

        # middle block
        # bin here
        self.conv2_2 = Binary_conv2d(layers[0], layers[1], 1) # padding for 1x1conv should be 0!
        # self.conv2_2 = nn.Conv2d(layers[0], layers[1], 1) # padding for 1x1conv should be 0!
        # relu here
        self.bn2_2 = Shiftbase_Batchnorm2d(layers[1], 4)
        
        # channelwise conv
        # bin here
        self.conv2_3 = Binary_conv2d(layers[1], layers[1], 1) # padding for 1x1conv should be 0!
        # self.conv2_3 = nn.Conv2d(layers[1], layers[1], 1) # padding for 1x1conv should be 0!
        # relu here
        self.bn2_3 = Shiftbase_Batchnorm2d(layers[1], 4)

        self.pooling2 = nn.MaxPool2d(2)
        
        '''Depthwise_conv_2'''
        # depthwise conv
        # bin here
        # pad here
        self.conv3_1 = Binary_conv2d(layers[1], layers[1], 4, 1, 0, groups=layers[1]//(2**group_dependency))
        # self.conv3_1 = nn.Conv2d(layers[1], layers[1], 4, 1, 0, groups=layers[1]//(2**group_dependency))
        # relu here
        self.bn3_1 = Shiftbase_Batchnorm2d(layers[1], 4)

        # middle block
        # bin here
        self.conv3_2 = Binary_conv2d(layers[1], layers[2], 1) # padding for 1x1conv should be 0!
        # self.conv3_2 = nn.Conv2d(layers[1], layers[2], 1) # padding for 1x1conv should be 0!
        # relu here
        self.bn3_2 = Shiftbase_Batchnorm2d(layers[2], 4)
        
        # channelwise conv
        # bin here
        self.conv3_3 = Binary_conv2d(layers[2], layers[2], 1) # padding for 1x1conv should be 0!
        # self.conv3_3 = nn.Conv2d(layers[2], layers[2], 1) # padding for 1x1conv should be 0!
        # relu here
        self.bn3_3 = Shiftbase_Batchnorm2d(layers[2], 4)

        # self.fc1 = Binary_linear(layers[2]*8*8, num_classes)
        self.fc1 = nn.Linear(layers[2]*8*8, num_classes)

    def forward(self, x):
        x = self.conv1(self.padder(x))
        x = self.bn1(x)
        x = self.pooling1(x)

        # 1st depthwise layer
        x = self.conv2_1(self.bin(self.padder(x)))
        x = self.bn2_1(x)

        x = self.conv2_2(self.bin(x))
        x = self.bn2_2(x)

        tx = self.conv2_3(self.bin(x))
        tx = self.bn2_3(tx)

        # x = self.conv2_3(self.bin(x))
        # x = self.bn2_3(x)

        x = self.pooling2(x)

        # 2nd depthwise layer
        x = self.conv3_1(self.bin(self.padder(x)))
        x = self.bn3_1(x)

        x = self.conv3_2(self.bin(x))
        x = self.bn3_2(x)

        tx = self.conv3_3(self.bin(x))
        tx = self.bn3_3(tx)

        # x = self.conv3_3(self.bin(x))
        # x = self.bn3_3(x)

        # x = x.clip(-20,20)
        # # Feature values here are accumulated by hundreds of +-1, but value is trivial here, sign is significant
        # # to simulate the deployment on scamp, x has only 20 values, so cut-off those larger than 20
        
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)

        return out


class MoreBiNet(nn.Module):
    def __init__(self, num_classes = 10, layers = [64,256], group_dependency = 1):
        super(MoreBiNet, self).__init__()
        # basic unit of a layer:
        # binarize--convolution--relu--batchnormalization
        self.bin = BinaryActivation.apply
        self.padder = Custom_Padding([1,2,1,2])
        self.conv1 = Binary_conv2d(3, layers[0], 4, 1, 0)
        self.bn1 = Shiftbase_Batchnorm2d(layers[0], 4)

        self.pooling1 = nn.MaxPool2d(2)
        
        '''Depthwise_conv_1'''
        # depthwise conv
        # bin here
        # pad here
        self.conv2_1 = Binary_conv2d(layers[0], layers[1], 4, 1, 0)
        self.bn2_1 = Shiftbase_Batchnorm2d(layers[1], 4)
        
        # channelwise conv
        # bin here
        self.conv2_3 = Binary_conv2d(layers[1], layers[1], 1) # padding for 1x1conv should be 0!
        # relu here
        self.bn2_3 = Shiftbase_Batchnorm2d(layers[1], 4)

        self.pooling2 = nn.MaxPool2d(2)
        
        '''Depthwise_conv_2'''
        # depthwise conv
        # bin here
        # pad here
        self.conv3_1 = Binary_conv2d(layers[1], layers[2], 4, 1, 0)
        # relu here
        self.bn3_1 = Shiftbase_Batchnorm2d(layers[2], 4)
        
        # channelwise conv
        # bin here
        self.conv3_3 = Binary_conv2d(layers[2], layers[2], 1) # padding for 1x1conv should be 0!
        # relu here
        self.bn3_3 = Shiftbase_Batchnorm2d(layers[2], 4)

        self.fc1 = Binary_linear(layers[2]*8*8, num_classes)

    def forward(self, x):
        x = self.conv1(self.padder(x))
        x = self.bn1(x)
        x = self.pooling1(x)

        # 1st depthwise layer
        x = self.conv2_1(self.bin(self.padder(x)))
        x = self.bn2_1(x)


        tx = self.conv2_3(self.bin(x))
        tx = self.bn2_3(tx)

        # x = self.conv2_3(self.bin(x))
        # x = self.bn2_3(x)

        x = self.pooling2(x)

        # 2nd depthwise layer
        x = self.conv3_1(self.bin(self.padder(x)))
        x = self.bn3_1(x)

        tx = self.conv3_3(self.bin(x))
        tx = self.bn3_3(tx)

        # x = self.conv3_3(self.bin(x))
        # x = self.bn3_3(x)

        # x = x.clip(-128,127)
        # # 但在这个卷积层，因为每个特征值是由一百多层的卷积核累加起来的，所以这里x的值会变成一百多，但是大小不重要，重要的是符号。
        # # 所以卷积结果必须存进areg里，如果要更加保真，我们需要给这层特征值加一个上限，比如说20。
        
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)

        return out