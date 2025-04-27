import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import wandb
from datetime import datetime
from collections import deque

# Utils for DDP setup
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN 
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from my_diffusion.model import DiffusionPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    eval_after = args['eval_after']
    ckpt_dir = args['ckpt_dir']
    load_dir = args['load_dir']
    if load_dir is None:
        load_dir = ckpt_dir
        print(f"load_dir not set. Setting to {ckpt_dir}")

    ckpt_name = args['ckpt_name']
    if ckpt_name is None:
        ckpt_name = 'best_policy.ckpt'
        print(f"ckpt_name not set. Setting to {ckpt_name}")

    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # setup device/DDP if using

    # only rank 0 process will print to stdout and save checkpoints
    is_logger = True
    device_id = 0

    if torch.cuda.is_available():
        use_distributed = args['use_distributed']
        if use_distributed:
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            global_rank = int(os.getenv("RANK", 0))
            world_size = torch.cuda.device_count()
            if world_size > 0:
                print(f"Dummy warning for single-gpu world.")
                dist.init_process_group(backend='nccl', rank=local_rank)
                # set a barrier until all processes reach this point
                dist.barrier(device_ids=[local_rank])
                device_id = dist.get_rank()
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device)
                # other rank processes don't log
                is_logger = (device_id == 0)
        else:
            device = 'cuda'


    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # task config wandb log
    wandb_log = args['wandb_log']


    # fixed parameters
    state_dim = task_config.get('state_dim', 14)
    lr_backbone = 1e-6 # from diffusion paper
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    elif policy_class == 'Diffusion':
        down_dims = [args['dim_feedforward'] * (2**k) for k in range(3)]
        n_obs_steps = 2
        n_act_steps = args['chunk_size'] // 2 #8 # chunk size
        horizon = args['chunk_size'] #16 
        drop_n_last_timesteps = horizon - n_act_steps - n_obs_steps + 1
        diffusion_step_embed_dim = args['hidden_dim'] # default 128
        clip_sample_range = args['clip_sample_range']
        policy_config = {
            'act_dim': state_dim,  # dimension of pos control space = same dim as pos
            'qpos_dim': state_dim, # dimension of position space
            'n_obs_steps': n_obs_steps, # number of input obs steps
            'n_action_steps': n_act_steps, # number of steps to actually use
            'horizon': horizon, # number of steps to predict args['chunk_dim']?
            'drop_n_last_frames': drop_n_last_timesteps,
            'diffusion_step_embed_dim': diffusion_step_embed_dim,
            'down_dims': down_dims, # downsample
            'img_shape': None,
            'num_train_timesteps': 100,
            'num_inference_steps': None,
            'clip_sample_range': clip_sample_range, # actions roughly between + - 3
            # vision model params
            'camera_names': camera_names,
            'lr_backbone': args['lr'] / 10,
            'img_shape': (3, 480,640),
            'n_groups': 8,
            'use_group_norm': False,
            'use_film_scale_modulation': True,

            # basic opt
            'lr': args['lr'],
            'weight_decay': args['weight_decay'],
            }
        
        '''
        act_dim,
        qpos_dim,
        img_shape,
        n_cameras,
        n_obs_steps=2,
        n_action_steps=8,
        horizon=16,
        drop_n_last_frames=7,
        down_dims=(512,1024,2048),
        diffusion_step_embed_dim=128,
        use_separate_backbone_per_camera=True, # for fair comparison
        lr=1e-5,
        lr_backbone=5e-5,
        weight_decay=1e-4,
        n_groups=8,
        use_group_norm=False,
        use_film_scale_modulation=True,
        '''

    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'wandb_log': wandb_log
    }

        
    if wandb_log and is_logger:
        # Use basename from args and add info from current run
        wandb_name = args.get('wandb_name', '')

        wandb_name += f'_pol=' + str(policy_class)
        wandb_name += f'_task=' + str(task_name)
        wandb_name += f'_cs=' + str(args['chunk_size'])

        now = datetime.now()
        wandb_name += f"_{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}"

        wandb.init(entity=args['wandb_entity'],
                project=args['wandb_project'],
                name=wandb_name)

        # update checkpoint directory with wandb run name
        if ckpt_dir is not None:
            print(f"WandB log: overwriting checkpoint directory to {wandb_name}")
            ckpt_dir += f"/{wandb_name}"
            config['ckpt_dir'] += f"/{wandb_name}"

        wandb.log(config, step=None)

    if is_eval:
        ckpt_names = [args['ckpt_name']]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, load_dir, ckpt_name, 
                                               use_distributed=use_distributed, device=device, is_logger=is_logger,
                                               save_episode=args['save_videos'])
            if wandb_log and is_logger:
                wandb.log({f'{ckpt_name}_eval_results': { 
                    'eval_success_rate': success_rate,
                        'eval_avg_return': avg_return
                        }
                       }, step=None)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            if is_logger:
                print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # drop 7 last frames from each episode for diffusion policy, otherwise don't drop any frames
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
                                                           drop_last_frames=policy_config.get('drop_n_last_frames', None),
                                                           horizon=policy_config.get('horizon', None),
                                                           n_obs_steps=policy_config.get('n_obs_steps', None),)
    if config['policy_class'] == "Diffusion":
        config['policy_config']['img_shape'] = stats['ds_meta']['observation.image']
    # Use distributed sampling to avoid reusing examples in DDP mode
    if use_distributed and dist.is_initialized():
        train_db = train_dataloader.dataset
        train_batch_size = train_dataloader.batch_size
        train_sampler = DistributedSampler(train_db, rank=dist.get_rank())
        train_dataloader = DataLoader(dataset=train_db,
                                batch_size=train_batch_size,
                                sampler=train_sampler)

        val_db = val_dataloader.dataset
        val_batch_size = val_dataloader.batch_size
        val_sampler = DistributedSampler(val_db, rank=dist.get_rank())
        val_dataloader = DataLoader(dataset=val_db,
                              batch_size=val_batch_size,
                              shuffle=False,
                              sampler=val_sampler)

    # save stats if and only if rank==0
    if is_logger and ckpt_dir is not None:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        # if checkpoint dir exists, shutdown
        else:
            raise FileExistsError(f"Checkpoint dir already exists at {ckpt_dir}. Shutting down.")

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, 
                            save_dir=ckpt_dir, use_distributed=use_distributed, device=device, is_logger=is_logger)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint if logging/rank==0
    if is_logger and ckpt_dir is not None:
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
    
    if eval_after:
        load_dir = ckpt_dir
        success_rate, avg_return = eval_bc(config, load_dir, ckpt_name='policy_best.ckpt', 
            use_distributed=use_distributed, device=device, is_logger=is_logger,
            save_episode=args['save_videos'])
        print(f"{success_rate=} {avg_return=}")
        if wandb_log:
            wandb.log({'success_rate_post_training': success_rate, 
                       'avg_return_post_training': avg_return})



def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(**policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer, scheduler = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer, scheduler


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_dir, ckpt_name, 
            use_distributed, device, is_logger, save_episode=True):
    set_seed(1000)
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    
    if use_distributed and dist.is_initialized():
        print('dist')
        policy.to(device)
        policy = DDP(policy, device_ids=[dist.get_rank()], output_device=dist.get_rank())
        state_dict = torch.load(ckpt_path, map_location=f"cuda:{dist.get_rank()}")
        loading_status = policy.module.load_state_dict(state_dict, strict=True)
    else:
        loading_status = policy.load_state_dict(torch.load(ckpt_path), strict=True)
        policy.to(device)

    print(loading_status)
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    if policy_class in ['ACT', 'CNNMLP']:
        query_frequency = policy_config['num_queries']
        if temporal_agg:
            query_frequency = 1
            num_queries = policy_config['num_queries']
    elif policy_class == 'Diffusion':
        query_frequency = policy_config['n_action_steps'] - 1
    else:
        raise NotImplementedError()

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
        if 'sim_pickup' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).to(device)

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).to(device)
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        
        with torch.inference_mode():
            for t in range(max_timesteps):
                
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                

                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos

                # stack n_obs_steps of img
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                elif config['policy_class'] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image) 
                    raw_action = all_actions[:, t % query_frequency]
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    if ckpt_dir is not None:
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write('\n\n')
            f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config, save_dir, device, is_logger, use_distributed,):
    num_epochs = config['num_epochs']
    seed = config['seed'] 
    
    if use_distributed and dist.is_initialized():
        seed += dist.get_rank()
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    wandb_log = config.get('wandb_log')

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    
    if use_distributed and dist.is_initialized():
        policy.model = DDP(policy.model, device_ids=[dist.get_rank()], output_device=dist.get_rank())

    optimizer, scheduler = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # define custom progress bar w/stats logged
    summary_string = "EMPTY"
    progress_bar = tqdm(range(num_epochs), postfix=summary_string)
    for epoch in progress_bar:
        
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        # Log val metrics to wandb
        if wandb_log:
            wandb.log({f"val_{k}": v for k, v in epoch_summary.items()}, commit=False, step=epoch)
        
        # create val summary string to append to progress bar
        val_summary_string = ''
        for k, v in epoch_summary.items():
            val_summary_string += f'{k}: {v.item():.3f} '

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step(epoch)

            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']

        # Log training epoch metrics to wandb
        if wandb_log:
            wandb.log({f"train_{k}": v for k, v in epoch_summary.items()}, commit=True, step=epoch)

        train_summary_string = ''
        for k, v in epoch_summary.items():
            train_summary_string += f'{k}: {v.item():.3f} '
        progress_bar.set_postfix_str(f"Train: {train_summary_string} | Val: {val_summary_string}")
        #print(summary_string)

        # save checkpoints way less often for memory.
        # normally i'd parametrize this as part of a training loop
        if epoch % 500 == 0 and save_dir is not None:
            ckpt_path = os.path.join(save_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, save_dir, seed)

    
    if save_dir is not None:
        ckpt_path = os.path.join(save_dir, f'policy_last.ckpt')
        torch.save(policy.state_dict(), ckpt_path)

    # save best policy, return best info no matter what
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    if save_dir is not None:
        ckpt_path = os.path.join(save_dir, f'best_policy_epoch_{best_epoch}_seed_{seed}.ckpt')
        torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    if save_dir is not None:
        plot_history(train_history, validation_history, num_epochs, save_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='dir to save checkpoints', required=False, default=None)
   
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # david's args
    parser.add_argument('--wandb_log', action='store', type=str, help='log to wandb', required=False, default=False)
    parser.add_argument('--wandb_entity', action='store', type=str, help='wandb username', required=False, default='dhpitt')
    parser.add_argument('--wandb_project', action='store', type=str, help='wandb username', required=False, default='')
    parser.add_argument('--wandb_name', action='store', type=str, help='wandb base run name', required=False, default='')

    parser.add_argument('--load_dir', action='store', type=str, help='dir to load checkpoint', required=False, default=None)
    parser.add_argument('--ckpt_name', action='store', type=str, help='name of checkpoint', required=False, default=None)
    parser.add_argument('--use_distributed', action='store', type=bool, help='whether to use ddp mode', required=False, default=False)
    parser.add_argument('--save_videos', action='store', type=bool, help='whether to save videos in eval', required=False, default=False)
    parser.add_argument('--clip_sample_range', action='store', type=float, help='magnitude at which to clip noise', required=False, default=None)
    parser.add_argument('--weight_decay', action='store', type=float, help='magnitude at which to clip noise', required=False, default=0.0001)
    parser.add_argument('--eval_after', action='store_true', help='whether to eval after training', required=False, default=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
