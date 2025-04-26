#!/bin/bash

for hidden_dim in 128 256; do

for batch_size in 8 16; do

MUJOCO_GL=egl poetry run python3 imitate_episodes.py \
	--task_name sim_pickup_scripted \
	--ckpt_dir ./local_data/sim_pickup_checkpts_diffusion_sweep \
	--policy_class Diffusion --kl_weight 10 --chunk_size 16 --hidden_dim $hidden_dim \
	--batch_size $batch_size --dim_feedforward 256 \
	--num_epochs 3000  --lr 1e-5 \
	--seed 0 \
	--wandb_log True --wandb_entity dhpitt \
	--wandb_project ultra-interview --wandb_name two_cam \
	--eval_after;

done;
done