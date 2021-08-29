## Model Architecture

- Predict `label` from **one classic model** 

## Dependencies

- python
- torch
- transformers


## Training & Evaluation



```bash

# random init embedding, lstm, max_pool, ce_loss

python src/classic_models/training/main.py --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --random_init_w2v --encoder lstm --aggregator max_pool --model_dir ./experiments/outputs/daguan/lstm_0815_1 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 6e-4 --learning_rate 20e-4 --classifier_learning_rate 20e-4 --warmup_steps 200 --max_seq_len 128 --hidden_dim 256 --embed_dim 256 --w2v_file resources/word2vec/dim_256/w2v.vectors --dropout_rate 0.2 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 5 

macro avg__f1-score__level_2 = 0.5070020675775782




# 训练好的 Word2vec embedding, lstm, max_pool, ce_loss

python src/classic_models/training/main.py --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --random_init_w2v --encoder lstm --aggregator max_pool --model_dir ./experiments/outputs/daguan/lstm_0815_2 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 8e-3 --learning_rate 1e-2 --classifier_learning_rate 1e-2 --warmup_steps 200 --max_seq_len 128 --hidden_dim 256 --embed_dim 256 --w2v_file resources/word2vec/dim_256/w2v.vectors --dropout_rate 0.2 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 5 

macro avg__f1-score__level_2 = 0.5013160507998644



# random init embedding, lstm, slf_attn_pool, ce_loss

python src/classic_models/training/main.py --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --random_init_w2v --encoder lstm --aggregator slf_attn_pool --model_dir ./experiments/outputs/daguan/lstm_0815_3 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 6e-4 --learning_rate 20e-4 --classifier_learning_rate 20e-4 --warmup_steps 200 --max_seq_len 128 --hidden_dim 256 --embed_dim 256 --w2v_file resources/word2vec/dim_256/w2v.vectors --dropout_rate 0.2 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 5 

macro avg__f1-score__level_2 = 0.


# 训练好的 Word2vec embedding, lstm, slf_attn_pool, ce_loss

python src/classic_models/training/main.py --data_dir ./datasets/phase_1/splits/fold_0 --label_file_level_1 datasets/phase_1/labels_level_1.txt --label_file_level_2 datasets/phase_1/labels_level_2.txt --task daguan --random_init_w2v --encoder lstm --aggregator slf_attn_pool --model_dir ./experiments/outputs/daguan/lstm_0815_4 --do_train --do_eval --train_batch_size 32 --num_train_epochs 50 --embeddings_learning_rate 8e-3 --learning_rate 1e-2 --classifier_learning_rate 1e-2 --warmup_steps 200 --max_seq_len 128 --hidden_dim 256 --embed_dim 256 --w2v_file resources/word2vec/dim_256/w2v.vectors --dropout_rate 0.2 --metric_key_for_early_stop "macro avg__f1-score__level_2" --logging_steps 400 --patience 5 

macro avg__f1-score__level_2 = 0.

```






