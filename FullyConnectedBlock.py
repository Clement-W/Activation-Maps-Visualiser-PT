from torch import nn


class FullyConnectedBlock(nn.Module):
    def __init__(self, nbInput, nbClasses):
        super(FullyConnectedBlock, self).__init__()
        self.linear = nn.Linear(nbInput, nbClasses)

    def forward(self, x):
        return self.linear(x)
