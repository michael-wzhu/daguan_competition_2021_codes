import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt


def vocab_process(data_dir, to_folder):

    # labels
    vocab_file_level_1 = os.path.join(to_folder, 'labels_level_1.txt')
    vocab_file_level_2 = os.path.join(to_folder, 'labels_level_2.txt')

    # label2freq
    label2freq_level_1_file = os.path.join(to_folder, "label2freq_level_1.json")
    label2freq_level_2_file = os.path.join(to_folder, "label2freq_level_2.json")

    with open(data_dir, 'r', encoding='utf-8') as f:
        vocab_level_1 = {}
        vocab_level_2 = {}

        for i, line in enumerate(f):
            if i == 0:
                continue

            label_ = line.strip().split(",")[-1]
            label_level_1 = label_.strip().split("-")[0]
            label_level_2 = label_

            if label_level_1 not in vocab_level_1:
                vocab_level_1[label_level_1] = 0
            vocab_level_1[label_level_1] += 1

            if label_level_2 not in vocab_level_2:
                vocab_level_2[label_level_2] = 0
            vocab_level_2[label_level_2] += 1

        json.dump(
            vocab_level_1,
            open(label2freq_level_1_file, "w", encoding="utf-8"),
        )
        json.dump(
            vocab_level_2,
            open(label2freq_level_2_file, "w", encoding="utf-8"),
        )

        vocab_level_1 = list(vocab_level_1.items())
        vocab_level_1 = sorted(vocab_level_1, key=lambda x: x[1], reverse=True)
        print("vocab_level_1: ", vocab_level_1)

        vocab_level_2 = list(vocab_level_2.items())
        vocab_level_2 = sorted(vocab_level_2, key=lambda x: x[1], reverse=True)
        print("vocab_level_2: ", vocab_level_2)

        # 画出柱状图，可视化样本数量
        x = [w[0] for w in vocab_level_2]
        y = [float(str(math.sqrt(w[1]))[:5]) for w in vocab_level_2]

        fig, ax = plt.subplots(figsize=(25, 17))
        ax.bar(
            x=x,  # Matplotlib自动将非数值变量转化为x轴坐标
            height=y,  # 柱子高度，y轴坐标
            width=0.8,  # 柱子宽度，默认0.8，两根柱子中心的距离默认为1.0
            align="center",  # 柱子的对齐方式，'center' or 'edge'
            color="grey",  # 柱子颜色
            edgecolor="red",  # 柱子边框的颜色
            linewidth=2.0  # 柱子边框线的大小
        )
        ax.set_title("Adjust Styles of Bar plot", fontsize=18)
        ax.set_xlabel("class name", fontsize=18)
        ax.set_ylabel("frequency", fontsize=18)

        # 一个常见的场景是：每根柱子上方添加数值标签
        # 步骤：
        # 1. 准备要添加的标签和坐标
        # 2. 调用ax.annotate()将文本添加到图表
        # 3. 调整样式，例如标签大小，颜色和对齐方式
        xticks = ax.get_xticks()
        for i in range(len(y)):
            xy = (xticks[i], y[i] * 1.03)
            s = str(y[i])
            ax.annotate(
                s=s,  # 要添加的文本
                xy=xy,  # 将文本添加到哪个位置
                fontsize=16,  # 标签大小
                color="blue",  # 标签颜色
                ha="center",  # 水平对齐
                va="baseline"  # 垂直对齐
            )

        plt.show()

        vocab_level_1 = [w[0] for w in vocab_level_1]
        vocab_level_2 = [w[0] for w in vocab_level_2]

        with open(vocab_file_level_1, "w", encoding="utf-8") as f_out:
            for lab in vocab_level_1:
                f_out.write(lab + '\n')

        with open(vocab_file_level_2, "w", encoding="utf-8") as f_out:
            for lab in vocab_level_2:
                f_out.write(lab + '\n')






if __name__ == "__main__":
    vocab_process(
        './datasets/phase_1/original/datagrand_2021_train.csv',
        "datasets/phase_1/"
    )
