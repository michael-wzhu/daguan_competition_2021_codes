# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict

from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

import sys
sys.path.insert(0, "./")



class SentenceIter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for i, fname in enumerate(os.listdir(self.dirname)):
            # if i > 5:
            #     continue
            print(fname)
            for line in open(os.path.join(self.dirname, fname), "r", encoding="utf-8"):
                yield line.strip()



def get_vocab_freq(sentence_iter, tokenizer):

    dict_vocab2freq = defaultdict(int)
    for i, sent in tqdm(enumerate(sentence_iter)):
        # if i > 5000:
        #     continue

        # print(sent)
        if not sent:
            continue

        tokens = tokenizer.tokenize(sent)
        for tok in tokens:
            dict_vocab2freq[tok] += 1

    return dict_vocab2freq


if __name__ == "__main__":

    # TOKENIZER
    tokenizer = BertTokenizer.from_pretrained(
        "resources/bert/chinese-bert-wwm-ext/"
    )

    corpus_folder = "datasets/news_corpus"
    sentence_iter = SentenceIter(corpus_folder)

    dict_vocab2freq = get_vocab_freq(sentence_iter, tokenizer)
    json.dump(
        dict_vocab2freq,
        open("src/bert_models/vocab_process/dict_vocab2freq_0819.json", "w", encoding="utf-8"),
        ensure_ascii=False,
    )

