# -*- coding: utf-8 -*-
import argparse
import os
import re

import torch
from transformers import BertConfig, BertTokenizer, BertModel, AlbertModel

import sys
sys.path.append("./")

from src.bert_models.models import ClsBERT
from src.classic_models.utils.model_utils import get_embedding_matrix_and_vocab

import numpy as np



def replace_albert_embeddings(pretrained_model_path,
                              new_model_path,
                              w2v_file,
                              model_type=None,
                              ):
    """
    Construct embedding matrix

    Args:
        pretrained_model_path : pretrained_model_path
        w2v_file : w2v_file
        skip_first_line : 是否跳过第一行
    Returns:
        None
    """

    # w2v: 先遍历一次，得到一个vocab list, 和向量的list
    vocab_list, vector_list = get_embedding_matrix_and_vocab(
        w2v_file, include_special_tokens=False
    )

    # 加载预训练模型部分
    MODEL_CLASSES = {
        'bert': (BertConfig, BertModel, BertTokenizer),
        'albert': (BertConfig, AlbertModel, BertTokenizer),
    }

    tokenizer = MODEL_CLASSES[model_type][2].from_pretrained(
        pretrained_model_path
    )
    config_class, model_class, _ = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(pretrained_model_path)
    model = model_class.from_pretrained(pretrained_model_path,
                                                  config=config,
                                                  )

    bert_embed_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()
    bert_vocab = tokenizer.get_vocab()
    print(type(bert_vocab))
    print(len(bert_vocab))
    print(bert_vocab["[PAD]"])
    # print(bert_embed_matrix[bert_vocab["[PAD]"]])

    # 构建新的vocab
    new_vocab_list, new_vector_list = [], []
    # 将[PAD], [UNK], [CLS], [SEP], [MASK] 的embedding加入
    for w in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        new_vocab_list.append(w)
        new_vector_list.append(bert_embed_matrix[bert_vocab[w]])

    for w_, vec_ in zip(vocab_list, vector_list):
        if not re.search("[0-9]", w_):
            print("non indexed word: ", w_)
            new_vocab_list.append(w_)
            new_vector_list.append(bert_embed_matrix[bert_vocab[w_]])

        else:
            new_vocab_list.append(w_)
            new_vector_list.append(vec_)

    assert len(new_vocab_list) == len(new_vector_list)

    vocab_file = os.path.join(new_model_path, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for w in new_vocab_list:
            f.write(w + "\n")

    config.vocab_size = len(new_vocab_list)
    config.save_pretrained(new_model_path)

    model.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.FloatTensor(new_vector_list))
    model.save_pretrained(new_model_path)


def replace_albert_embeddings_random(pretrained_model_path,
                              new_model_path,
                              w2v_file,
                              embedding_dim=128,
                                     model_type="bert"):
    """
    Construct embedding matrix

    Args:
        pretrained_model_path : pretrained_model_path
        w2v_file : w2v_file
        skip_first_line : 是否跳过第一行
    Returns:
        None
    """

    # w2v: 先遍历一次，得到一个vocab list, 和向量的list
    vocab_list, _ = get_embedding_matrix_and_vocab(w2v_file, include_special_tokens=False)

    # 加载预训练模型部分

    MODEL_CLASSES = {
        'bert': (BertConfig, BertModel, BertTokenizer),
        'albert': (BertConfig, AlbertModel, BertTokenizer),
    }

    tokenizer = MODEL_CLASSES[model_type][2].from_pretrained(
        pretrained_model_path
    )
    config_class, model_class, _ = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(pretrained_model_path)
    model = model_class.from_pretrained(pretrained_model_path,
                                                  config=config,
                                                  )

    bert_embed_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()
    bert_vocab = tokenizer.get_vocab()
    print(type(bert_vocab))
    print(len(bert_vocab))
    print(bert_vocab["[PAD]"])
    # print(albert_embed_matrix[albert_vocab["[PAD]"]])

    # 构建新的vocab
    new_vocab_list, new_vector_list = [], []
    # 将[PAD], [UNK], [CLS], [SEP], [MASK] 的embedding加入
    for w in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
        new_vocab_list.append(w)
        new_vector_list.append(bert_embed_matrix[bert_vocab[w]])

    for w_ in vocab_list:
        if not re.search("[0-9]", w_):
            print("non indexed word: ", w_)
            new_vocab_list.append(w_)
            new_vector_list.append(bert_embed_matrix[bert_vocab[w_]])

        else:
            new_vocab_list.append(w_)
            new_vector_list.append(
                (np.random.randn(embedding_dim).astype(np.float32) * 0.2).tolist()
            )

    assert len(new_vocab_list) == len(new_vector_list)

    vocab_file = os.path.join(new_model_path, "vocab.txt")
    with open(vocab_file, "w", encoding="utf-8") as f:
        for w in new_vocab_list:
            f.write(w + "\n")

    config.vocab_size = len(new_vocab_list)
    config.save_pretrained(new_model_path)

    model.embeddings.word_embeddings.weight = torch.nn.Parameter(
        torch.FloatTensor(new_vector_list)
    )
    model.save_pretrained(new_model_path)


if __name__ == "__main__":

    # for bert base, 随机初始化矩阵替换embedding
    pretrained_model_path = "resources/bert/chinese-bert-wwm-ext"
    w2v_file = "resources/word2vec/dim_256/w2v.vectors"
    new_model_path = "resources/bert/chinese-bert-wwm-ext_embedding_replaced_random"
    replace_albert_embeddings_random(pretrained_model_path,
                                     new_model_path,
                                     w2v_file,
                                     embedding_dim=768,
                                     model_type="bert",
                                     )

    # bert base： 训练好的w2v替换bert原本的embedding
    pretrained_model_path = "resources/bert/chinese-bert-wwm-ext"
    w2v_file = "resources/word2vec/dim_768/w2v.vectors"
    new_model_path = "resources/bert/chinese-bert-wwm-ext_embedding_replaced_w2v"
    replace_albert_embeddings(pretrained_model_path,
                              new_model_path,
                              w2v_file,
                              model_type="bert",
                              )

    # 模型： MacBERT；
    # # for MacBERT base, 随机初始化矩阵替换embedding
    # pretrained_model_path = "resources/bert/chinese-macbert-base"
    # w2v_file = "resources/word2vec/dim_256/w2v.vectors"
    # new_model_path = "resources/bert/chinese-macbert-base_embedding_replaced_random"
    # replace_albert_embeddings_random(pretrained_model_path,
    #                                  new_model_path,
    #                                  w2v_file,
    #                                  embedding_dim=768,
    #                                  model_type="bert",
    #                                  )

    # # for chinese roberta wwm ext base, 随机初始化矩阵替换embedding
    # pretrained_model_path = "resources/bert/chinese-roberta-wwm-ext"
    # w2v_file = "resources/word2vec/dim_128_sg_0_hs_1_epochs_5/w2v.vectors"
    # new_model_path = "resources/bert/chinese-roberta-wwm-ext_embedding_replaced_random"
    # replace_albert_embeddings_random(pretrained_model_path,
    #                                  new_model_path,
    #                                  w2v_file,
    #                                  embedding_dim=768,
    #                                  model_type="bert",
    #                                  )
    #
    # # for chinese roberta wwm ext large, 随机初始化矩阵替换embedding
    # pretrained_model_path = "resources/bert/chinese-roberta-wwm-ext-large"
    # w2v_file = "resources/word2vec/dim_128_sg_0_hs_1_epochs_5/w2v.vectors"
    # new_model_path = "resources/bert/chinese-roberta-wwm-ext-large_embedding_replaced_random"
    # replace_albert_embeddings_random(pretrained_model_path,
    #                                  new_model_path,
    #                                  w2v_file,
    #                                  embedding_dim=1024,
    #                                  model_type="bert",
    #                                  )