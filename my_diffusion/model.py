from collections import deque

from torch import nn

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from torch import Tensor, nn
import torchvision.transforms as transforms

class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """


    def __init__(
        self,
        act_dim,
        qpos_dim,
        img_shape,
        camera_names,
        n_obs_steps=2,
        n_action_steps=8,
        horizon=16,
        drop_n_last_frames=7,
        num_train_timesteps=100,
        num_inference_steps=None,
        clip_sample_range=None,
        down_dims=(512,1024,2048),
        diffusion_step_embed_dim=128,
        use_separate_backbone_per_camera=True, # for fair comparison
        lr=1e-5,
        lr_backbone=1e-6,
        weight_decay=1e-6,
        n_groups=8,
        use_group_norm=False,
        use_film_scale_modulation=True,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        
        # queues are populated during rollout of the policy, 
        # they contain the n latest observations and actions
        self._qpos_queue = None
        self._action_queue = None
        self._image_queue = None
        
        self.act_dim = act_dim
        self.qpos_dim = qpos_dim
        self.img_shape = img_shape
        self.camera_names = camera_names
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.drop_n_last_frames = drop_n_last_frames
        self.down_dims = down_dims
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.use_separate_backbone_per_camera = use_separate_backbone_per_camera
        

        self.diffusion = DiffusionModel(act_dim=act_dim,
                                        qpos_dim=qpos_dim,
                                        img_shape=img_shape,
                                        n_obs_steps=n_obs_steps,
                                        n_action_steps=n_action_steps,
                                        horizon=horizon,
                                        num_train_timesteps=num_train_timesteps,
                                        num_inference_steps=num_inference_steps,
                                        clip_sample_range=clip_sample_range,
                                        drop_n_last_frames=drop_n_last_frames,
                                        down_dims=down_dims,
                                        diffusion_step_embed_dim=diffusion_step_embed_dim,
                                        n_groups=n_groups,
                                        use_group_norm=use_group_norm,
                                        use_film_scale_modulation=use_film_scale_modulation,
                                        camera_names=camera_names,
                                        use_separate_backbone_per_camera=True)
        
        for n, p in self.diffusion.named_parameters():
            if "backbone" in n:
                p.requires_grad = False
                
        param_dicts = [
                {"params": [p for n, p in self.diffusion.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.diffusion.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": 0.,
                },
            ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                    weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=3000,

        )

        self.reset()

    def configure_optimizers(self):
        return self.optimizer, self.scheduler

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._qpos_queue = deque(maxlen=self.n_obs_steps)
        self._action_queue = deque(maxlen=self.n_action_steps)
        self._image_queue = deque(maxlen=self.n_obs_steps)

    @torch.no_grad
    def select_action(self, qpos, images, actions=None, is_pad=None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        images = normalize(images)

        # print(f"{qpos=}")
        # print(f"{qpos.shape=}")

        
        '''if self.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.image_features], dim=-4
            )'''
        # Note: It's important that this happens after stacking the images into a single key.
        if actions is not None:
            self._action_queue = populate_single_queue(self._action_queue, actions)
        self._qpos_queue = populate_single_queue(self._qpos_queue, qpos)
        self._image_queue = populate_single_queue(self._image_queue, images)
        
        if len(self._action_queue) == 0:
            # stack n latest observations from the queue
            qpos_batch = torch.stack(list(self._qpos_queue), dim=1)
            img_batch = torch.stack(list(self._image_queue), dim=2) #b, n_cam, c, h, w --> b, n_cam, n_obs, c, h, w 
            a_hat = self.diffusion.generate_actions(qpos_batch, img_batch)

            self._action_queue.extend(a_hat.transpose(0, 1))

        action = self._action_queue.popleft()
        return action

    def forward(self, qpos, images, actions=None, is_pad=None) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""

        ## Training/val
        if actions is not None:
            batch_size, horizon, act_dim = actions.shape
            actions = actions[:, :self.horizon]
            if horizon < self.horizon:
                new_actions = torch.zeros(batch_size, self.horizon, act_dim)
                new_actions[:,:horizon] = actions
                new_actions[:, horizon:] = actions[:, -1] # clone the last frame up to self.drop_n_last timesteps
            loss = self.diffusion.compute_loss(qpos, images, actions, is_pad)
            loss_dict = {'mse': loss, 'loss': loss}
            # no output_dict so returning None
            return loss_dict
        ## inference
        else:
            a_hat = self.select_action(qpos, images, actions, is_pad)
            return a_hat


class DiffusionModel(nn.Module):
    def __init__(self, 
                act_dim,
                qpos_dim,
                img_shape,
                camera_names,
                n_obs_steps=2,
                n_action_steps=8,
                horizon=16,
                num_train_timesteps=100,
                num_inference_steps=None,
                clip_sample_range=None,
                drop_n_last_frames=7,
                do_mask_loss_for_padding=False,
                vision_backbone='resnet18',
                pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
                spatial_softmax_keypoints=32,
                use_group_norm=False,
                crop_shape=None,
                crop_random=False,
                down_dims=(512,1024,2048),
                diffusion_step_embed_dim=128,
                n_groups=8,
                kernel_size=5,
                use_film_scale_modulation=True, 
                use_separate_backbone_per_camera=True, # for fair comparison
                 ):
        super().__init__()

        self.act_dim=act_dim
        self.qpos_dim=qpos_dim
        self.qpos_dim=qpos_dim
        self.img_shape=img_shape
        self.camera_names=camera_names
        self.n_obs_steps=n_obs_steps
        self.n_action_steps=n_action_steps
        self.horizon=horizon
        self.drop_n_last_frames = drop_n_last_frames
        self.prediction_type = "epsilon" # hardcoded diffusion

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = qpos_dim
        self.use_separate_backbone_per_camera = use_separate_backbone_per_camera
        num_images = len(self.camera_names)
        if self.use_separate_backbone_per_camera:
            encoders = [
                DiffusionRgbEncoder(vision_backbone=vision_backbone,
                                    image_shape=img_shape,
                                    spatial_softmax_num_keypoints=spatial_softmax_keypoints,
                                    use_group_norm=use_group_norm,
                                    crop_is_random=crop_random,
                                    crop_shape=crop_shape,
                                    pretrained_weights=pretrained_backbone_weights
                                    ) for _ in range(num_images)
                                    ]
            self.backbone = nn.ModuleList(encoders)
            global_cond_dim += encoders[0].feature_dim * num_images
        else:
            self.backbone = DiffusionRgbEncoder(vision_backbone=vision_backbone,
                                    image_shape=img_shape,
                                    spatial_softmax_num_keypoints=spatial_softmax_keypoints,
                                    use_group_norm=use_group_norm,
                                    crop_is_random=crop_random,
                                    crop_shape=crop_shape,
                                    pretrained_weights=pretrained_backbone_weights
                                    )
            global_cond_dim += self.backbone.feature_dim * num_images

        self.unet = DiffusionConditionalUnet1d(diffusion_step_embed_dim=diffusion_step_embed_dim,
                                               act_dim=act_dim,
                                               down_dims=down_dims,
                                               n_groups=n_groups,
                                               kernel_size=kernel_size,
                                               use_film_scale_modulation=use_film_scale_modulation, 
                                               global_cond_dim=global_cond_dim * n_obs_steps)


        # if clip sample range provided, clip=True
        clip_sample = clip_sample_range is not None
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range, # actions unit normal min -2.8 max 2.8 ish
            prediction_type="epsilon",
        )

        if num_inference_steps is None:
            self.num_inference_steps = num_train_timesteps
        else:
            self.num_inference_steps = num_inference_steps

        self.do_mask_loss_for_padding = do_mask_loss_for_padding

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.horizon, self.act_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, qpos, images) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = qpos.shape[:2]
        global_cond_feats = [qpos]
        #print(f"{images.shape=}")
        # Extract image features.
        if self.use_separate_backbone_per_camera:
            # images come in b, n_cameras, n_obs, c, h, w == one image per action
            # Combine batch and sequence dims while rearranging to make the camera index dimension first.
            images = einops.rearrange(images, "b n s ... -> n (b s) ...")
            # Combine batch and sequence dims while rearranging to make the camera index dimension first.
            img_features = []
            for i, backbone in enumerate(self.backbone):
                img_features.append(backbone(images[i])) # grab the n-th camera's images
            img_features = torch.cat(img_features,dim=0) # n, b, (shape)
            # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        else:
            raise NotImplementedError()
            '''# Combine batch, sequence, and "which camera" dims before passing to shared encoder.
            img_features = self.backbone(
                einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
            )
            # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )'''
        global_cond_feats.append(img_features)


        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, qpos, images) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        # print(f"in generate, {qpos.shape=}")
        batch_size, n_obs_steps, _ = qpos.shape
        assert n_obs_steps == self.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(qpos, images)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.n_action_steps
        actions = actions[:, start:end]
        #print(f"{actions.shape=}")
        return actions

    def compute_loss(self, qpos, images, action, is_pad) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        n_obs_steps = qpos.shape[1]
        horizon = action.shape[1]

        assert horizon == self.horizon
        assert n_obs_steps == self.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(qpos, images)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = action
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.prediction_type == "epsilon":
            target = eps
        elif self.prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.do_mask_loss_for_padding:
            if is_pad is None:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~is_pad
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, vision_backbone='resnet18', 
                 image_shape=(3,480,640),
                 spatial_softmax_num_keypoints=32,
                 use_group_norm=False,
                 crop_shape=None, 
                 crop_is_random=None, 
                 pretrained_weights=None):
        super().__init__()
        # Set up optional preprocessing.
        if crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, vision_backbone)(
            weights=pretrained_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if use_group_norm:
            if pretrained_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        
        dummy_shape_c_h_w = crop_shape if crop_shape is not None else image_shape[-3:] # c, h, w
        dummy_shape = (1, *dummy_shape_c_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)
        self.feature_dim = spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, 
                 diffusion_step_embed_dim,
                 act_dim,
                 down_dims,
                 kernel_size,
                 n_groups,
                 use_film_scale_modulation,
                 global_cond_dim: int):
        super().__init__()

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        # act_dim replaced action_feature.shape[0]
        in_out = [(act_dim, down_dims[0])] + list(
            zip(down_dims[:-1], down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    down_dims[-1], down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], act_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
    

### stolen lerobot utils

def populate_single_queue(queue, elem):
    if len(queue) != queue.maxlen:
        # initialize by copying the first observation several times until the queue is full
        while len(queue) != queue.maxlen:
            queue.append(elem)
    else:
        # add latest observation to the queue
        queue.append(elem)
    return queue


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)