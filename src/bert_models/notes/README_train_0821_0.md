## Model Architecture

- Predict `label` from **one classic model** 

## Dependencies

- python
- torch
- transformers

## Dataset

|       | Train  | Dev | Test |  Labels (level 1) | Labels (level 2) |
| ----- | ------ | --- | ---- |  ----------- |     ---- |
| daguan  | 12k  | 2k | 6k  |  10    |   35  |

- The number of labels are given by the dataset.

## Training & Evaluation


### BERT model

#### 随机初始化 embedding

```bash

#

# bert, bert_pooler, ce_loss,
python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/chinese-bert-wwm-ext_embedding_replaced_random --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/bert_0821_0 --do_train --do_eval --train_batch_size 16 --num_train_epochs 50 --embeddings_learning_rate 0.4e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 12 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json 

# on dev
macro avg__f1-score__level_2 = 0.x





```


#### 用词汇对照，将脱敏数据集转化为明文

```bash

#

# bert, bert_pooler, ce_loss,
python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/chinese-bert-wwm-ext --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/bert_0821_0 --do_train --do_eval --train_batch_size 16 --num_train_epochs 50 --embeddings_learning_rate 0.4e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 12 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json 

# on dev
macro avg__f1-score__level_2 = 0.x





```


