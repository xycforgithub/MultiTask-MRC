#!/usr/bin/env bash

# Train on SQuAD+NewsQA+MARCO, with sample reweighting
python train.py --multitask_data_path ../data/mtmrc/mt_snm/ --valid_batch_size 16 --highway_dropout 0.1 --ema_gamma 0.999 --train_datasets squad,newsqa,marco --extra_score_file nm_lm_scores.json

# Train on SQuAD+NewsQA+MARCO, with ELMo and sample reweighting and multi gpu
python train.py --multitask_data_path ../data/mtmrc/mt_snm/ --train_datasets squad,newsqa,marco --dataset_config_id 1 --highway_dropout 0.0 --ema_gamma 0.995 --elmo_config_id 1 --elmo_l2 0.0001 --add_elmo --multi_gpu --extra_score_file nm_lm_scores.json
