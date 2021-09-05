import argparse
import logging
import os

import pandas as pd


#### Just some code to print debug information to stdout
import random

import sys
sys.path.insert(0, "./")



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,)

label_list_level_1 = [
    label.strip() for label in open(
        "./datasets/phase_1/labels_level_1.txt", 'r', encoding='utf-8')
]
label_list_level_2 = [
    label.strip() for label in open(
        "./datasets/phase_1/labels_level_2.txt", 'r', encoding='utf-8')
]


def get_label_name2sents(train_data_path):
    dict_label_name2sents = {}

    with open(train_data_path, 'r', encoding='utf8') as fIn:
        for i, row in enumerate(fIn):
            row = row.strip()
            if not row:
                continue
            row = row.split(",")
            # print(row)

            sent = row[1].strip()
            label_name = row[2].strip()

            if label_name not in dict_label_name2sents:
                dict_label_name2sents[label_name] = set()

            dict_label_name2sents[label_name].add(sent)

    dict_label_name2sents_new = {}
    for label_name, sents in dict_label_name2sents.items():
        dict_label_name2sents_new[label_name] = list(sents)

    return dict_label_name2sents_new



parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_path", default=None, type=str,
    help="path to the dataset;"
)
parser.add_argument(
    "--nli_dataset_path", default=None, type=str,
    help="path to the nli dataset;"
)
parser.add_argument(
    "--sampling_times", default=4, type=int,
    help="times of sampling;"
)

args = parser.parse_args()

logging.info("Read in train dataset")
train_data_path = os.path.join(args.dataset_path, "train.txt")
dict_label_name2sents_train = get_label_name2sents(train_data_path)


logging.info("Generate nli train dataset with hard negatives")
train_samples = []
for i in range(args.sampling_times):
    label_0, label_1 = random.sample(label_list_level_2, 2)   #
    anchor, pos = random.sample(list(dict_label_name2sents_train[label_0]), 2)
    neg = random.choice(list(dict_label_name2sents_train[label_1]))

    train_samples.append(
        {
            "sent0": anchor,
            "sent1": pos,
            "hard_neg": neg
        }
    )

df_train_samples = pd.DataFrame(
    train_samples
)
df_train_samples.to_csv(
    os.path.join(args.nli_dataset_path, "nli_for_simcse.csv"),
    index=False,
    sep="\t"
)