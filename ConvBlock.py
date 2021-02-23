from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, nb_in, nb_out):
        super(ConvBlock, self).__init__()
        self.convolution = nn.Conv2d(
            in_channels=nb_in,
            out_channels=nb_out,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.ReLU = nn.ReLU()
        self.MaxPooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.convolution(x)
        x = self.ReLU(x)
        x = self.MaxPooling(x)
        return x
