# -*- coding: utf-8 -*-
"""
@File: max_pool.py
@Copyright: 2019 Michael Zhu
@License：the Apache License, Version 2.0
@Author：Michael Zhu
@version：
@Date：
@Desc: 
"""

import torch
# from allennlp.nn import util
from torch import nn

from src.classic_models.utils.model_utils import replace_masked_values


class AvgPoolerAggregator(nn.Module):
    """
    A ``AvgPoolerAggregator`` is a avg pooling layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, ) -> None:
        super(AvgPoolerAggregator, self).__init__()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_max_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """
        if mask is not None:
            # Simple Pooling layers
            input_tensors = replace_masked_values(
                input_tensors, mask.unsqueeze(2), 0
            )

        tokens_avg_pooled = torch.mean(input_tensors, 1)

        return tokens_avg_pooled
