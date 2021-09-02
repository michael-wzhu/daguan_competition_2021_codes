
## 数据处理：
```bash

# 统计标签信息
src/data_proc/label_vocab_process.py

# 句长统计
src/data_proc/sample_length_stats.py

# 数据集拆分
src/data_proc/split_datasets.py

```

## 词频对应

本题采用脱敏文本，所以我们需要将词汇与开源BERT词汇对应，得到明文数据

```bash

# 从一个corpus中得到词频统计
src/bert_models/vocab_process/get_vocab_freq_from_corpus.py

# 得到词频对应，并将数据集转为明文
src/bert_models/vocab_process/get_vocab_mapping.py

```


## 训练示例：
```bash

nohup python src/bert_models/training/main.py 
    --model_type nezha 
    --model_name_or_path resources/nezha/NEZHA-Base 
    --data_dir ./datasets/phase_1/splits/fold_0_bertvocab 
    --label_file_level_1 datasets/phase_1/labels_level_1.txt 
    --label_file_level_2 datasets/phase_1/labels_level_2.txt 
    --task daguan --aggregator bert_pooler 
    --model_dir ./experiments/outputs/daguan/nezha_0821_0 
    --do_train --do_eval 
    --train_batch_size 32 
    --num_train_epochs 50 
    --embeddings_learning_rate 0.5e-4 
    --encoder_learning_rate 0.5e-4 
    --classifier_learning_rate 5e-4 
    --warmup_steps 400 
    --max_seq_len 132 
    --dropout_rate 0.15 
    --metric_key_for_early_stop "macro avg__f1-score__level_2" 
    --logging_steps 400 
    --patience 6 
    --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json 
    --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json 
    --processor_sep "\t" 
> ./experiments/logs/nezha_0821_0.log &


```




## Results
模型结果记录: 采用5折交叉验证的平均分

|      模型描述      | dev macro-F1 | 
| ------------ |  -------------- | 
|          官方baseline：   word2vec + bilstm                |        0.392         |
|          随机word2vec + bilstm +  max-pool               |        0.491  |                        |
|              - + slf_attn_pool               |        0.49267002199706494  |                        |
|          预训练word2vec + bilstm +  max-pool               |       0.5013160507998644         |                    |
|           -  +  slf_attn_pool               |       0.4972518334336293         |  
|          BERT-base + 随机初始化embedding               |       0.505         |  
|             - + 词汇表词频对应               |       0.515         |  
|          BERT-wwm-ext + 词汇表词频对应               |       0.524         |  
|            -   多种pooling操作一起使用               |       0.528         |  
|          - + sample weights               |       0.535         |  
|          - + ce + NTXENT loss               |       0.533         |  
|          NEZHA-base-wwm +  词汇表词频对应                |       0.535         |  
|          - +  ce + NTXENT loss (系数0.1, gamma 0.5)                |       0.538         |  
|          - +  ce + NTXENT loss (系数0.5, gamma 0.5)                |       0.530         |  
|          - +  ce + NTXENT loss (系数0.5, gamma 0.07)                |       0.526         |  
|          - +  multi-sample dropout (0.4, num=4, sum)                |       0.543         |  
|          - +  multi-sample dropout (num=4, average)                |       0.xxx?         |  
|          - +  multi-sample dropout (num=8, average)                |       0.xxx?         |  