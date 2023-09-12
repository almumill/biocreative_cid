#!/bin/bash

batch_sizes=("10 16 32 64")
learning_rates=("1e-05 3e-05 5e-05")
weighted_loss=("--weighted_loss")

for lr in ${learning_rates[@]}; do
  for batch_size in ${batch_sizes[@]}; do
    for weighted_loss_flag in ${weighted_loss[@]}; do
      # Set directory names for organized models
      if [ "${weighted_loss_flag}" = "--weighted_loss" ]; then
        wl_dir_name='weighted_loss'
        echo "Running models/batch_size_${batch_size}/lr_${lr}/${wl_dir_name}"
        python3 model_es.py cdr models/batch_size_${batch_size}/lr_${lr}/${wl_dir_name} cdr_data/cdr_train_data.tsv cdr_data/cdr_dev_data.tsv --lr ${lr} --weighted_loss --validation_only --save_checkpoints --batch_size ${batch_size} --num_epochs 20
      else
        wl_dir_name='unweighted_loss'
        echo "Running models/batch_size_${batch_size}/lr_${lr}/${wl_dir_name}"
        echo python3 model_es.py cdr models/batch_size_${batch_size}/lr_${lr}/${wl_dir_name} cdr_data/cdr_train_data.tsv cdr_data/cdr_dev_data.tsv --lr ${lr}  --validation_only --save_checkpoints --batch_size ${batch_size} --num_epochs 20
        python3 model_es.py cdr models/batch_size_${batch_size}/lr_${lr}/${wl_dir_name} cdr_data/cdr_train_data.tsv cdr_data/cdr_dev_data.tsv --lr ${lr}  --validation_only --save_checkpoints --batch_size ${batch_size} --num_epochs 20
      fi

    done
  done
done
