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


#### Nezha model, with class weights / sample weights


```bash

# Nezha base, bert_pooler, loss_fct_name=ce, --use_class_weights
CUDA_VISIBLE_DEVICES="0" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0828_3 --do_train --do_eval --train_batch_size 64 --num_train_epochs 50 --embeddings_learning_rate 0.7e-4 --encoder_learning_rate 0.7e-4 --classifier_learning_rate 7e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --loss_fct_name ce --use_class_weights > ./experiments/logs/nezha_0828_3.log &

> 0.55

# Nezha base, bert_pooler, loss_fct_name=ce, --use_weighted_sampler
CUDA_VISIBLE_DEVICES="2" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0828_4 --do_train --do_eval --train_batch_size 64 --num_train_epochs 50 --embeddings_learning_rate 0.7e-4 --encoder_learning_rate 0.7e-4 --classifier_learning_rate 7e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --loss_fct_name ce --use_weighted_sampler > ./experiments/logs/nezha_0828_4.log &

0.549936791137791

```



#### Nezha model, with different loss functions

1) 用词汇对照，将脱敏数据集转化为明文
2) contrastive loss: 样本与样本之间距离根据标签调整

```bash



# Nezha base, bert_pooler, loss_fct_name=ce, contrastive_loss=None
CUDA_VISIBLE_DEVICES="0" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0828_0 --do_train --do_eval --train_batch_size 64 --num_train_epochs 50 --embeddings_learning_rate 0.7e-4 --encoder_learning_rate 0.7e-4 --classifier_learning_rate 7e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --loss_fct_name ce > ./experiments/logs/nezha_0828_0.log &

# on dev
macro avg__f1-score__level_2 = 0.5402011803141538



# Nezha base, bert_pooler, loss_fct_name=focal, contrastive_loss=None
CUDA_VISIBLE_DEVICES="1" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0828_1 --do_train --do_eval --train_batch_size 64 --num_train_epochs 50 --embeddings_learning_rate 0.7e-4 --encoder_learning_rate 0.7e-4 --classifier_learning_rate 7e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --loss_fct_name focal --focal_loss_gamma 0.3 > ./experiments/logs/nezha_0828_1.log &

# on dev
macro avg__f1-score__level_2 = 0.5295988600871874



# Nezha base, bert_pooler, loss_fct_name=dice, contrastive_loss=None
CUDA_VISIBLE_DEVICES="2" nohup python src/bert_models/training/main.py --model_type nezha --model_name_or_path resources/nezha/NEZHA-Base --data_dir ./datasets/phase_1/splits/fold_0_bertvocab --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --aggregator bert_pooler --model_dir ./experiments/outputs/daguan/nezha_0828_2 --do_train --do_eval --train_batch_size 64 --num_train_epochs 50 --embeddings_learning_rate 0.7e-4 --encoder_learning_rate 0.7e-4 --classifier_learning_rate 7e-4 --warmup_steps 200 --max_seq_len 132 --dropout_rate 0.15 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 200 --patience 6 --label2freq_level_1_dir datasets/phase_1/label2freq_level_1.json --label2freq_level_2_dir datasets/phase_1/label2freq_level_2.json --processor_sep "\t" --loss_fct_name dice > ./experiments/logs/nezha_0828_2.log &

# on dev
macro avg__f1-score__level_2 = 0.5584147006320698






```





