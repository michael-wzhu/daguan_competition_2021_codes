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


### 更多 BERT model


#### Nezha model

用词汇对照，将脱敏数据集转化为明文

```bash

#

# Nezha base, bert_pooler, ce_loss,
nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0821_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" > ./experiments/logs/nezha_0821_0.log &

# on dev
macro avg__f1-score__level_2 = 0.5430708862032628


# Nezha base, bert_pooler, ce_loss,
nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator dr_pooler --model_dir ./experiments/outputs/daguan/nezha_0821_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" > ./experiments/logs/nezha_0821_1.log &


# on dev
macro avg__f1-score__level_2 = 0.5288330751661092


```



#### BERT-wwm-ext model

用词汇对照，将脱敏数据集转化为明文

```bash

#

# Nezha base, bert_pooler, ce_loss,
CUDA_VISIBLE_DEVICES="2" python src/bert_models/training/main.py --model_type bert --model_name_or_path resources/bert/chinese-bert-wwm-ext --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/bert_wwm_0825_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 50 --patience 12 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t"

# on dev
macro avg__f1-score__level_2 = 0.526899613822347





```

