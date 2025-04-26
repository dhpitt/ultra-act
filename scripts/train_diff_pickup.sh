#!/bin/bash

poetry run python3 imitate_episodes.py \
	--task_name sim_pickup_scripted \
	--ckpt_dir ./local_data/sim_pickup_checkpts_diffusion \
	--policy_class Diffusion --kl_weight 10 --chunk_size 16 --hidden_dim 512 --batch_size 8 --dim_feedforward 256 \
	--num_epochs 2000  --lr 5e-5 \
	--seed 0 \
	--wandb_log True --wandb_entity dhpitt \
	--wandb_project diffpol --wandb_name two_cam \
