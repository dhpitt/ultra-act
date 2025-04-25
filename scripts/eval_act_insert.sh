#!/bin/bash

MUJOCO_GL=egl poetry run python3 imitate_episodes.py \
	--task_name sim_insertion_scripted \
	--load_dir ./local_data/sim_insertion_checkpts/replicate_intro_policy=ACT_task=sim_insertion_scripted_4-24-17-20-47 \
	--ckpt_name ultra_best.ckpt \
	--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
	--num_epochs 0  --lr 1e-5 \
	--seed 0 \
	--eval \

	#--temporal_agg \