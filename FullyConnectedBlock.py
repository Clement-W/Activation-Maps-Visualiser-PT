from torch import nn


class FullyConnectedBlock(nn.Module):
    def __init__(self, nbInput, nbClasses):
        super(FullyConnectedBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nbInput, nbClasses)
            # nn.Dropout()
        )

    def forward(self, x):
        return self.block(x)
