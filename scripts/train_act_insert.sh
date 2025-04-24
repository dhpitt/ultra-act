#!/bin/bash

poetry run python3 imitate_episodes.py \
	--task_name sim_insertion_scripted \
	--ckpt_dir ./local_data/sim_insertion_checkpts \
	--policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
	--num_epochs 2000  --lr 3e-5 \
	--wandb_log True --wandb_entity dhpitt \
	--wandb_project ultra-intervew --wandb_name smaller_chunks \
	--seed 0