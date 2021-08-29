import torch
from torch import nn

from src.classic_models.utils.model_utils import get_sinusoid_encoding_table


class SinusoidPositionalEmbedding(torch.nn.Module):
    """
    SinusoidPositionalEmbedding:

    Parameters
    ----------
    """

    def __init__(self, max_len=512, embed_dim=300) -> None:
        super(SinusoidPositionalEmbedding, self).__init__()

        self.encoder_position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                max_len,
                embed_dim,
                padding_idx=0
            ),
            freeze=True
        )

    def forward(self, input_pos_tensors: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_pos_tensors : (batch_size, num_tokens).
        Returns
        -------

        """

        return self.encoder_position_enc(input_pos_tensors)


class LearnedPositionalEmbedding(torch.nn.Module):
    """
    SinusoidPositionalEmbedding:

    Parameters
    ----------
    """

    def __init__(self, max_len=512, embed_dim=300, positional_embedding=None) -> None:
        super(LearnedPositionalEmbedding, self).__init__()

        if positional_embedding:
            self.encoder_position_enc = positional_embedding
        else:
            self.encoder_position_enc = nn.Embedding(
                max_len,
                embed_dim,
                padding_idx=0
            )

    def forward(self, input_pos_tensors: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_pos_tensors : (batch_size, num_tokens).
        Returns
        -------

        """

        return self.encoder_position_enc(input_pos_tensors)