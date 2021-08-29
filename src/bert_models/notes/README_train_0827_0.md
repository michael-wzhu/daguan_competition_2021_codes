## Model Architecture

- Predict `label` from **one classic model** 

## Dependencies

- python
- torch
- transformers
- pytorch_metric_learning

## Dataset

|       | Train  | Dev | Test |  Labels (level 1) | Labels (level 2) |
| ----- | ------ | --- | ---- |  ----------- |     ---- |
| daguan  | 12k  | 2k | 6k  |  10    |   35  |

- The number of labels are given by the dataset.

## Training & Evaluation


### 更多 BERT model


#### Nezha model, with contrastive loss

1) 用词汇对照，将脱敏数据集转化为明文
2) contrastive loss: 样本与样本之间距离根据标签调整

```bash

#

# Nezha base, bert_pooler, ce_loss, ntxent_loss=v1
CUDA_VISIBLE_DEVICES="2" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0821_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --ntxent_loss v1 > ./experiments/logs/nezha_0821_0.log &

# on dev
macro avg__f1-score__level_2 = 0.xxx



# Nezha base, bert_pooler, ce_loss, ntxent_loss=v2
CUDA_VISIBLE_DEVICES="0" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0827_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --ntxent_loss v2 > ./experiments/logs/nezha_0827_0.log &




# Nezha base, bert_pooler, ce_loss, ntxent_loss=v2, use_ntxent_loss_only
CUDA_VISIBLE_DEVICES="1" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0827_2 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --ntxent_loss v2 --use_ntxent_loss_only > ./experiments/logs/nezha_0827_2.log &


# Nezha base, bert_pooler, ce_loss, ntxent_loss=v3
CUDA_VISIBLE_DEVICES="2" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0827_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --ntxent_loss v3 > ./experiments/logs/nezha_0827_1.log &

CUDA_VISIBLE_DEVICES="2" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0827_6 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --ntxent_loss v3 --contrastive_temperature 0.07 > ./experiments/logs/nezha_0827_6.log &


# Nezha base, bert_pooler, ce_loss, ntxent_loss=v2
CUDA_VISIBLE_DEVICES="0" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0827_5 --do_train --do_eval --train_batch_size 8 --num_train_epochs 50 --embeddings_learning_rate 0.5e-4 --encoder_learning_rate 0.5e-4 --classifier_learning_rate 5e-4 --warmup_steps 400 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --ntxent_loss v2 --ntxent_loss_weight 1.0 > ./experiments/logs/nezha_0827_5.log &



```





