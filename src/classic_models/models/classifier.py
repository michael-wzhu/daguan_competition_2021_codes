
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

