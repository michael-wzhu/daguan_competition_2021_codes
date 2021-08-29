import torch
from torch import nn
import torch.nn.functional as F


class Identity(nn.Module):

    def __init__(self, dropout=0.1):
        super(Identity, self).__init__()
        self.dropout = dropout

    def forward(self, x, mask=None):
        if self.dropout:
            x = F.dropout(x, self.dropout, self.training,)
        return x


if __name__ == "__main__":

    op = Identity(dropout=0.5)
    x = torch.randn((3, 12))

    x = op(x)
    print(x)