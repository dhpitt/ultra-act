#!/bin/bash

MUJOCO_GL=egl poetry run python3 imitate_episodes.py \
	--task_name sim_pickup_scripted \
	--policy_class Diffusion --kl_weight 10 --chunk_size 16 --hidden_dim 256 \
	--batch_size 8 --dim_feedforward 256 \
	--num_epochs 10  --lr 1e-5 \
	--clip_sample_range 3.0 \
	--seed 0 \
	--eval_after;
