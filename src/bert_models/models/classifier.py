# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)   # [batch_size, hidden_dim]

        return self.linear(x)


class MultiSampleClassifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(MultiSampleClassifier, self).__init__()
        self.args = args

        self.linear = nn.Linear(input_dim, num_labels)

        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(args.dropout_rate) for _ in range(self.args.dropout_num)]
        )

    def forward(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)

            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits

        # 相加还是求平均？
        if self.args.ms_average:
            logits = logits / self.args.dropout_num

        return logits