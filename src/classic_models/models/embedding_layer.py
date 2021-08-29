
import numpy as np
import torch

import torch.nn as nn

from src.classic_models.utils.model_utils import get_embedding_matrix_and_vocab
from src.classic_models.modules.positional_embedding import SinusoidPositionalEmbedding


class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super(EmbeddingLayer, self).__init__()

        self.args = args

        # 模型维数
        self.embed_dim = args.embed_dim

        # 加载embedding：从word2vec的文件中加载
        vocab_list, vector_list = get_embedding_matrix_and_vocab(
            args.w2v_file, skip_first_line=True)
        self.vocab_list = vocab_list

        assert self.embed_dim == len(vector_list[0])
        assert len(vocab_list) == len(vector_list)

        self.w2v_matrix = np.asarray(vector_list)

        # 初始化embedding
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(
                len(self.vocab_list),
                self.embed_dim,
            )
        else:
            self.word_embedding = nn.Embedding(
                len(self.vocab_list),
                self.embed_dim,
            ).from_pretrained(torch.FloatTensor(self.w2v_matrix), freeze=False)

        self.positional_embedding = SinusoidPositionalEmbedding(
            max_len=args.max_seq_len,
            embed_dim=self.embed_dim
        )

        # 正则部分
        # embedding dropout
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.embed_dim)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        device = input_ids.device

        inputs_embeds = self.word_embedding(input_ids)
        # inputs_embeds_freezed = self.word_embedding_freezed(input_ids)

        # inputs_embeds = (inputs_embeds + inputs_embeds_freezed) / 2

        # 加上positional embedding
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        position_embeddings =self.positional_embedding(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings





