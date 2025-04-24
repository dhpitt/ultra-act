IDX=10
NUM_EPISODES=50

record_sim:
	MUJOCO_GL=egl poetry run python3 record_sim_episodes.py \
	--task_name sim_insertion_scripted \
	--dataset_dir ./local_data/sim_insertion_scripted \
	--num_episodes ${NUM_EPISODES} 

visualize_recording:
	poetry run python3 visualize_episodes.py --dataset_dir ./local_data/sim_insertion_scripted --episode_idx ${IDX} 

train:
	poetry run python3 imitate_episodes.py \
	--task_name sim_insertion_scripted \
	--ckpt_dir ./local_data/sim_insertion_checkpts \
	--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0

eval:
	MUJOCO_GL=egl poetry run python3 imitate_episodes.py \
	--task_name sim_insertion_scripted \
	--ckpt_dir ./local_data/sim_insertion_checkpts \
	--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 --temporal_agg \
	--seed 0 \
	--eval

install:
	poetry install
	poetry run pip install torch
