from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, nb_in, nb_out):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=nb_in,
                out_channels=nb_out,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(nb_out),
            nn.MaxPool2d(kernel_size=2)
            # nn.Dropout2d() # The dropout will cause a lower training accuracy
        )

    def forward(self, x):
        return self.block(x)
