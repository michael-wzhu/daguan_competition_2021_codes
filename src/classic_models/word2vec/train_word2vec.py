# -*- coding: utf-8 -*-

import os

from gensim.models import Word2Vec

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print(fname)
            for line in open(os.path.join(self.dirname, fname), "r", encoding="utf-8"):
                yield line.split()


def train_w2v_model(model_save_dir, sentences: MySentences, ):
    print('Start...')
    model = Word2Vec(
        sentences,
        sg=0,
        hs=1,
        vector_size=128,
        window=12,
        min_count=1,
        workers=2,
        epochs=5,
    )

    print(model.max_final_vocab)

    model.wv.save_word2vec_format(model_save_dir, binary=False)
    print("Finished!")


if __name__ == "__main__":
    model_save_dir = "./ppts/第一课-开营/codes/resources/word2vec/dim_256/w2v.vectors"
    sentences = MySentences("./datasets/无监督数据/txt_format/")
    # sentences = MySentences("./datasets/无监督数据/jsonline")

    train_w2v_model(model_save_dir, sentences)




