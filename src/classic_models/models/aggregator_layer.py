import torch.nn as nn

from src.classic_models.modules.avg_pool import AvgPoolerAggregator
from src.classic_models.modules.child_dynamic_routing import DynamicRoutingAggregator
from src.classic_models.modules.max_pool import MaxPoolerAggregator
from src.classic_models.modules.self_attn_pool import SelfAttnAggregator



class AggregatorLayer(nn.Module):

    def __init__(self, args, aggregator_name=None
    ):
        super(AggregatorLayer, self).__init__()
        self.args = args
        self.d_model = args.hidden_dim
        self.aggregator_op_name = aggregator_name

        self.aggregator_op = None
        if self.aggregator_op_name == "slf_attn_pooler":
            attn_vector = nn.Linear(
                self.d_model,
                1,
            )
            self.aggregator_op = SelfAttnAggregator(
                self.d_model,
                attn_vector=attn_vector,
            )
        elif self.aggregator_op_name == "dr_pooler":
            cap_num_ = 4  # capsule 大小
            iter_num_ = 3  # 迭代次数
            shared_fc_ = nn.Linear(
                self.d_model,
                self.d_model
            )
            self.aggregator_op = DynamicRoutingAggregator(
                self.d_model,
                cap_num_,
                int(self.d_model / cap_num_),
                iter_num_,
                shared_fc=shared_fc_,
                device=args.device
            )
        elif self.aggregator_op_name == "max_pooler":
            self.aggregator_op = MaxPoolerAggregator()
        else:
            self.aggregator_op = AvgPoolerAggregator()

    def forward(self, input_tensors, mask=None):
        output = self.aggregator_op(input_tensors, mask)

        return output














