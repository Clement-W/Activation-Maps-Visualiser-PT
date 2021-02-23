from torch import nn
from FullyConnectedBlock import FullyConnectedBlock
from ConvBlock import ConvBlock


"""
class Network(nn.Module):

    def __init__(self, kernels, nb_classes=10):
        super(Network,self).__init__()
        layers=[]
        layers.append(ConvBlock(1,kernels[0])) # 28*28*1 -> 14*14*16
        layers.append(ConvBlock(kernels[0],kernels[1])) # 14*14*16 -> 7*7*32
        layers.append(ConvBlock(kernels[1],kernels[2])) # 7*7*32 -> 3*3*64
        layers.append(nn.Flatten()) # 576
        layers.append(FullyConnectedBlock(3*3*kernels[-1],nb_classes))
 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


~One other way to do create the network~"""

class Network(nn.Module):
    def __init__(self, kernels, nb_classes=10):
        super(Network,self).__init__()
        
        self.conv1 = ConvBlock(1,kernels[0]) # 28*28*1 -> 14*14*16
        self.conv2 = ConvBlock(kernels[0],kernels[1]) # 14*14*16 -> 7*7*32
        self.conv3 = ConvBlock(kernels[1],kernels[2]) # 7*7*32 -> 3*3*64
        self.flatten = nn.Flatten() # 576
        self.fclayer = FullyConnectedBlock(3*3*kernels[-1],nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten(x)
        x = self.fclayer(x)

        return x