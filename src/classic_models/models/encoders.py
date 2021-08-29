
import torch.nn as nn

from src.classic_models.modules.child_rnns import RnnEncoder
from src.classic_models.modules.child_sep_conv import ChildSepConv


class TextCnnEncoder(nn.Module):
    def __init__(self, args):
        super(TextCnnEncoder, self).__init__()
        self.args = args

        # 四个卷积核
        self.ops = nn.ModuleList()
        for kernel_size in [1, 3, 5, 7]:
            op_ = ChildSepConv(
                self.args.embed_dim,
                self.args.hidden_dim,
                kernel_size,
            )
            self.ops.append(op_)

        # 正则操作
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, input_tensors=None,
                attention_mask=None,
                position_ids=None,
                **kwargs):

        tmp_outputs = []
        for i, op in enumerate(self.ops):
            input_tensors_conv = op(input_tensors)
            tmp_outputs.append(input_tensors_conv)

        output_tensors = sum(tmp_outputs)
        output_tensors = self.dropout(output_tensors)
        # output_tensors = self.LayerNorm(output_tensors + input_tensors)
        output_tensors = self.LayerNorm(output_tensors)

        return output_tensors


class BiLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(BiLSTMEncoder, self).__init__()
        self.args = args

        # 一个LSTM
        self.op = RnnEncoder(
            self.args.embed_dim,
            self.args.hidden_dim,
            rnn_name="lstm",
            bidirectional=True
        )

        # 正则操作
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.args.hidden_dim)

    def forward(self, input_tensors=None,
                attention_mask=None,
                position_ids=None,
                **kwargs):

        output_tensors = self.op(input_tensors)
        output_tensors = self.dropout(output_tensors)
        # output_tensors = self.LayerNorm(output_tensors + input_tensors)
        output_tensors = self.LayerNorm(output_tensors)

        return output_tensors


