import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, 
                 dataset_dir, 
                 camera_names, 
                 norm_stats, 
                 drop_last_frames=None, 
                 n_obs_steps=None, 
                 horizon=None,
                 importance_sampling=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.drop_last_frames = drop_last_frames
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.importance_sampling = importance_sampling

        if self.horizon is not None:
            assert self.n_obs_steps is not None
        if self.n_obs_steps is not None:
            assert self.horizon is not None
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                if self.n_obs_steps is None:
                    start_ts = np.random.choice(episode_len)
                else:
                    # sample from 1 to end - drop_last_frames
                    sample_range = episode_len - self.drop_last_frames - 1
                    if self.importance_sampling:
                        probs = np.arange(sample_range)
                        linear_elbow = sample_range // 2
                        # linearly increase probability until halfway through dataset
                        probs[linear_elbow:] = probs[linear_elbow]
                        probs += probs[linear_elbow] # bump up all values
                        p = probs / probs.sum()

                        start_ts = 1 + np.random.choice(sample_range, p=p)
                    else:
                        # uniformly sample from start to end
                        start_ts = 1 + np.random.choice(sample_range) # start at 1
            # get observation at start_ts only
            if self.n_obs_steps is None:
                qpos = root['/observations/qpos'][start_ts]
            else:  
                qpos = root['/observations/qpos'][start_ts:start_ts+self.n_obs_steps]
            #qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                if self.n_obs_steps is None:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts:start_ts+self.n_obs_steps]
            # get all actions after and including last obs timestep

            action_start_ts = start_ts
            if self.n_obs_steps is not None:
                action_start_ts += self.n_obs_steps - 1
            if is_sim:
                action = root['/action'][action_start_ts:]
                action_len = episode_len - action_start_ts
            else:
                action = root['/action'][max(0, action_start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, action_start_ts - 1) # hack, to make timesteps more aligned
            
            '''if self.horizon is not None:
                action = action[:self.horizon] # hack, to make timesteps more aligned
                action_len = self.horizon # hack, to make timesteps more aligned'''

        self.is_sim = is_sim

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
    
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()


        # channel last
        if self.n_obs_steps is None:
            image_data = torch.einsum('k h w c -> k c h w', image_data) # ncams, c, h, e
        else:
            image_data = torch.einsum('k n h w c -> k n c h w', image_data) # n_cams, n_obs, c, h, w

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        '''print(f"{image_data.shape=}")
        print(f"{action_data.shape=}")
        print(f"{qpos_data.shape=}")'''
        '''if self.drop_last_frames is not None:
            action_data = action_data[:-self.drop_last_frames]
            qpos_data = qpos_data[:-self.drop_last_frames]
            is_pad = is_pad[:-self.drop_last_frames]'''

        '''print(f"{image_data.min()=}{image_data.max()=}")
        print(f"{qpos_data.min()=}{qpos_data.max()=}")
        print(f"{action_data.min()=}{action_data.max()=}")'''
        #raise ValueError()
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, 
              num_episodes, 
              camera_names, 
              batch_size_train, 
              batch_size_val, 
              drop_last_frames, 
              horizon, n_obs_steps, 
              importance_sampling):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, 
                                    drop_last_frames=drop_last_frames,n_obs_steps=n_obs_steps, horizon=horizon, 
                                    importance_sampling=importance_sampling)
    images, obs, action, _ = train_dataset[0]
    
    # store shapes of one batch for constructing diffusionpolicy
    norm_stats["ds_meta"] = dict({
        "observation.image": images.shape,
        "observation.state": obs.shape,
        "action": action.shape
    })
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, 
                                  drop_last_frames=drop_last_frames,n_obs_steps=n_obs_steps, horizon=horizon,
                                  importance_sampling=importance_sampling)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
