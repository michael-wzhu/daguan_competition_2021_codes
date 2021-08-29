# -*- coding: utf-8 -*-


import torch
# from allennlp.nn import util
from torch import nn

from src.classic_models.utils.model_utils import masked_softmax, weighted_sum


class SelfAttnAggregator(nn.Module):
    """
    A ``SelfAttnAggregator`` is a self attn layers.  As a
    :class:`SelfAttnAggregator`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, output_dim,
                 attn_vector=None) -> None:
        super(SelfAttnAggregator, self).__init__()

        self.output_dim = output_dim

        self.attn_vector = None
        if attn_vector:
            self.attn_vector = attn_vector
        else:
            self.attn_vector = nn.Linear(
                self.output_dim,
                1
            )

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (batch_size, num_tokens, input_dim).
        mask : sentence mask, (batch_size, num_tokens).
        Returns
        -------
        input_self_attn_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self.attn_vector(
            input_tensors
        ).squeeze(2)
        self_weights = masked_softmax(self_attentive_logits, mask)
        input_self_attn_pooled = weighted_sum(input_tensors, self_weights)

        return input_self_attn_pooled
