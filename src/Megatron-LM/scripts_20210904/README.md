

### 准备工作

1) 装有cuda的服务器;
2) 安装docker;
3) 获取NVIDIA官方镜像
```bash
docker pull nvcr.io/nvidia/pytorch:20.12-py3
```



### 准备预训练数据

因为我们的数据格式与原项目的格式略有不同，我们稍微修改下python预处理代码

```bash

cd src/Megatron-LM

python scripts_20210904/preprocess_data_ours.py --input ../../datasets/pretraining_data/jsonline/datagrand_2021_unlabeled_data.json --output-prefix ../../experiments/outputs/pretraining/data_0904/unlabeled_data --vocab ../../resources/daguan_bert_base_v3/steps_120k/vocab.txt --dataset-impl mmap --tokenizer-type BertWordPieceLowerCase --split-sentences --workers 2

```


### 多卡分布式训练

```bash

./scripts_20210904/pretrain_bert_distributed.sh

```