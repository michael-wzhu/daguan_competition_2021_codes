from torch import nn


class Zero(nn.Module):

    def __init__(self, ):
        super(Zero, self).__init__()

    def forward(self, x, mask=None):
        x = x.mul(0.)
        return x