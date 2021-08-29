
import json
import os
from collections import defaultdict

import numpy as np

import pandas as pd

import sys
sys.path.append("./")
from src.classic_models.utils.model_utils import get_embedding_matrix_and_vocab

if __name__ == "__main__":
    # 先得到本次比赛的词频排序
    # w2v: 先遍历一次，得到一个vocab list, 和向量的list
    w2v_file = "resources/word2vec/dim_768/w2v.vectors"
    daguan_freq_rank2vocab, _ = get_embedding_matrix_and_vocab(
        w2v_file, include_special_tokens=False
    )
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("，"))
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("。"))
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("！"))
    daguan_freq_rank2vocab.pop(daguan_freq_rank2vocab.index("？"))

    # 数据集中出现的词汇：
    vocab_in_tasks = defaultdict(int)
    df_train = pd.read_csv("./datasets/phase_1/splits/fold_0/train.txt", header=None)
    df_train.columns = ["id", "text", "label"]
    df_val = pd.read_csv("./datasets/phase_1/splits/fold_0/dev.txt", header=None)
    df_val.columns = ["id", "text", "label"]
    df_test = pd.read_csv("./datasets/phase_1/splits/fold_0/test.txt", header=None)
    df_test.columns = ["id", "text", ]

    for df_ in [df_train, df_val, df_test]:
        for text in df_['text']:
            for char in text.split(" "):
                vocab_in_tasks[char] += 1

    print(vocab_in_tasks)
    print(len(vocab_in_tasks))
    # for v in vocab_in_tasks:
    #     print(v, daguan_freq_rank2vocab.index(v))

    # 得到一个中文BERT词汇频率的排序
    counts = json.load(open('src/bert_models/vocab_process/vocab_freq/counts.json', encoding="utf-8"))
    del counts["[CLS]"]
    del counts["[SEP]"]
    # del counts["[UNK]"]
    # del counts["[PAD]"]
    # del counts["[MASK]"]

    del counts["，"]
    del counts["。"]
    del counts["！"]
    del counts["？"]
    # print(counts)

    token_dict = {}
    with open("resources/bert/chinese-bert-wwm-ext/vocab.txt", encoding="utf-8") as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    del token_dict["[CLS]"]
    del token_dict["[SEP]"]
    del token_dict["[UNK]"]
    del token_dict["[PAD]"]
    del token_dict["[MASK]"]

    del token_dict["，"]
    del token_dict["。"]
    del token_dict["！"]
    del token_dict["？"]

    # print(token_dict)
    list_bert_vocab2freqs = [
        (i, counts.get(i, 0)) for i, j in
        sorted(token_dict.items(), key=lambda s: s[1])
    ]
    print(list_bert_vocab2freqs)
    list_bert_vocab2freqs = sorted(
        list_bert_vocab2freqs,
        key=lambda x: x[1],
        reverse=True
    )
    list_bert_vocab2freq_rank = [w[0] for w in list_bert_vocab2freqs]
    print(list_bert_vocab2freq_rank)


    dict_daguan_vocab2bert_vocab = {}
    count_unused = 0
    for v in vocab_in_tasks:
        if v in daguan_freq_rank2vocab:
            v_freq_rank_in_daguan = daguan_freq_rank2vocab.index(v)

            if v_freq_rank_in_daguan < len(list_bert_vocab2freq_rank):
                v_in_bert = list_bert_vocab2freq_rank[v_freq_rank_in_daguan]
            else:
                v_in_bert = "[unused%d]" % (count_unused + 1)
                count_unused += 1
                print(v_in_bert, v)

            dict_daguan_vocab2bert_vocab[v] = v_in_bert

        else:
            print("not included in daguan_freq_rank2vocab: ", v)

    dict_daguan_vocab2bert_vocab["，"] = "，"
    dict_daguan_vocab2bert_vocab["。"] = "。"
    dict_daguan_vocab2bert_vocab["！"] = "！"
    dict_daguan_vocab2bert_vocab["？"] = "？"

    print(dict_daguan_vocab2bert_vocab)

    for df_ in [df_train, df_val, df_test]:
        for text in df_['text']:
            text = text.split(" ")
            text_new = [dict_daguan_vocab2bert_vocab[w] for w in text]
            text_new = " ".join(text_new)
            assert "\t" not in text_new

            print(text_new)
            # for char in text.split(" "):
            #     vocab_in_tasks[char] += 1

    for i in range(len(df_train)):
        text = df_train['text'][i]
        text = text.split(" ")
        text_new = [dict_daguan_vocab2bert_vocab[w] for w in text]
        text_new = " ".join(text_new)
        df_train.loc[i, "text"] = text_new
    df_train.to_csv(
        os.path.join(
            "datasets/phase_1/splits/fold_0_bertvocab/",
            "train.txt"
        ),
        index=False,
        sep="\t",
        header=None,
        encoding="utf-8",
    )

    for i in range(len(df_val)):
        text = df_val['text'][i]
        text = text.split(" ")
        text_new = [dict_daguan_vocab2bert_vocab[w] for w in text]
        text_new = " ".join(text_new)
        df_val.loc[i, "text"] = text_new
    df_val.to_csv(
        os.path.join(
            "datasets/phase_1/splits/fold_0_bertvocab/",
            "dev.txt"
        ),
        index=False,
        sep="\t",
        header=None,
        encoding="utf-8",
    )
    for i in range(len(df_test)):
        text = df_test['text'][i]
        text = text.split(" ")
        text_new = [dict_daguan_vocab2bert_vocab[w] for w in text]
        text_new = " ".join(text_new)
        df_test.loc[i, "text"] = text_new
    df_test.to_csv(
        os.path.join(
            "datasets/phase_1/splits/fold_0_bertvocab/",
            "test.txt"
        ),
        index=False,
        sep="\t",
        header=None,
        encoding="utf-8",
    )



