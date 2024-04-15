import os
import numpy as np
import pandas as pd
import xarray as xr

import torch

torch.random.seed()
np.random.seed(0)

import warnings

warnings.filterwarnings("ignore")

import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import random
from functools import partial
from itertools import repeat
from typing import Callable
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
import math
import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
from timm.models.convnext import ConvNeXtBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
from timm.models.vision_transformer import Block as ViTBlock
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

import torch.utils.data
import numpy as np
import cv2
import glob

import os.path as osp
import tempfile
import re
import shutil
import sys
import ast
from importlib import import_module
import argparse


def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2**32 - 1))


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


def resize_func(x):
    return cv2.resize(x, (128, 128), interpolation=cv2.INTER_LINEAR)


def resize_back_func(x):
    return cv2.resize(x, (161, 161), interpolation=cv2.INTER_LINEAR)


def chunk_time(ds):
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(
        dim,
        dim,
        kernel_size=kernel,
        padding=(kernel - 1) // 2,
        bias=bias,
        groups=dim)


def UniformerSubBlock(embed_dims,
                      mlp_ratio=4.,
                      drop=0.,
                      drop_path=0.,
                      init_value=1e-6,
                      block_type='Conv'):
    """Build a block of Uniformer."""

    assert block_type in ['Conv', 'MHSA']
    if block_type == 'Conv':
        return CBlock(
            dim=embed_dims,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path)
    else:
        return SABlock(
            dim=embed_dims,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value)


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def create_parser():
    parser = argparse.ArgumentParser(description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument(
        '--dist',
        action='store_true',
        default=False,
        help='Whether to use distributed training (DDP)')
    parser.add_argument(
        '--display_step',
        default=10,
        type=int,
        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--ex_name', '-ex', default='era5_tau_sub', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=False,
        help=
        'Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)'
    )
    parser.add_argument(
        '--torchscript',
        action='store_true',
        default=False,
        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        default=False,
        help='Whether to set different seeds for different ranks')
    parser.add_argument(
        '--fps',
        action='store_true',
        default=False,
        help='Whether to measure inference speed (FPS)')
    parser.add_argument(
        '--empty_cache',
        action='store_true',
        default=True,
        help='Whether to empty cuda cache after GPU training')
    parser.add_argument(
        '--find_unused_parameters',
        action='store_true',
        default=False,
        help='Whether to find unused parameters in forward during DDP training'
    )
    parser.add_argument(
        '--broadcast_buffers',
        action='store_false',
        default=True,
        help='Whether to set broadcast_buffers to false during DDP training')
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto_resume',
        action='store_true',
        default=False,
        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='Only performs testing')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=False,
        help=
        'whether to set deterministic options for CUDNN backend (reproducable)'
    )
    parser.add_argument(
        '--launcher',
        default='none',
        type=str,
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help='job launcher for distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--port',
        type=int,
        default=29500,
        help='port only works when launcher=="slurm"')

    # dataset parameters
    parser.add_argument(
        '--batch_size', '-b', default=8, type=int, help='Training batch size')
    parser.add_argument(
        '--val_batch_size',
        '-vb',
        default=8,
        type=int,
        help='Validation batch size')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument(
        '--data_root', default='/home/convlstm_predict/OpenSTL-master')
    # parser.add_argument('--event', default='0ll8.tif')
    parser.add_argument(
        '--dataname',
        '-d',
        default='era5',
        type=str,
        help='Dataset name (default: "mmnist")')
    parser.add_argument(
        '--pre_seq_length',
        default=None,
        type=int,
        help='Sequence length before prediction')
    parser.add_argument(
        '--aft_seq_length',
        default=None,
        type=int,
        help='Sequence length after prediction')
    parser.add_argument(
        '--total_length',
        default=None,
        type=int,
        help='Total Sequence length for prediction')
    parser.add_argument(
        '--use_augment',
        action='store_true',
        default=False,
        help='Whether to use image augmentations for training')
    parser.add_argument(
        '--use_prefetcher',
        action='store_true',
        default=False,
        help='Whether to use prefetcher for faster data loading')
    parser.add_argument(
        '--drop_last',
        action='store_true',
        default=False,
        help='Whether to drop the last batch in the val data loading')

    # method parameters
    parser.add_argument(
        '--method',
        '-m',
        default='SimVP',
        type=str,
        choices=[
            'ConvLSTM', 'convlstm', 'CrevNet', 'crevnet', 'DMVFN', 'dmvfn',
            'E3DLSTM', 'e3dlstm', 'MAU', 'mau', 'MIM', 'mim', 'PhyDNet',
            'phydnet', 'PredNet', 'prednet', 'PredRNN', 'predrnn', 'PredRNNpp',
            'predrnnpp', 'PredRNNv2', 'predrnnv2', 'SimVP', 'simvp', 'TAU',
            'tau'
        ],
        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument(
        '--config_file',
        '-c',
        default='config.py',
        type=str,
        help='Path to the default config file')
    parser.add_argument(
        '--model_type',
        default=None,
        type=str,
        help='Name of model for SimVP (default: None)')
    parser.add_argument(
        '--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.0,
        help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Whether to allow overwriting the provided config file with args')

    # Training parameters (optimizer)
    parser.add_argument(
        '--epoch', '-e', default=100, type=int, help='end epochs')
    parser.add_argument(
        '--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument(
        '--opt',
        default='adam',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adam"')
    parser.add_argument(
        '--opt_eps',
        default=None,
        type=float,
        metavar='EPSILON',
        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--clip_mode',
        type=str,
        default='norm',
        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # Training parameters (scheduler)
    parser.add_argument(
        '--sched',
        default='onecycle',
        type=str,
        metavar='SCHEDULER',
        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument(
        '--lr_k_decay',
        type=float,
        default=1.0,
        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='warmup learning rate (default: 1e-5)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument(
        '--final_div_factor',
        type=float,
        default=1e4,
        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument(
        '--warmup_epoch',
        type=int,
        default=0,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--decay_epoch',
        type=float,
        default=100,
        metavar='N',
        help='epoch interval to decay LR')
    parser.add_argument(
        '--decay_rate',
        '--dr',
        type=float,
        default=0.1,
        metavar='RATE',
        help='LR decay rate (default: 0.1)')
    parser.add_argument(
        '--filter_bias_and_bn',
        type=bool,
        default=False,
        help='LR decay rate (default: 0.1)')

    return parser


def load_config(filename: str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config


def update_config(args, config, exclude_keys=list()):
    """update the args dict with a new config"""
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if args.get(k, False):
            if args[k] != config[k] and k not in exclude_keys:
                print(f'overwrite config key -- {k}: {config[k]} -> {args[k]}')
            else:
                args[k] = config[k]
        else:
            args[k] = config[k]
    return args


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=False,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=True,
                  worker_seeding='all'):
    sampler = None
    if distributed and not isinstance(dataset,
                                      torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(
                    dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle
        and (not isinstance(dataset, torch.utils.data.IterableDataset))
        and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers)
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    return loader


def load_data(batch_size,
              val_batch_size,
              data_root,
              training_time,
              num_workers=4,
              distributed=False,
              use_augment=False,
              use_prefetcher=False,
              drop_last=False,
              **kwargs):

    weather_dataroot = data_root

    test_set = WeatherBenchDataset(
        weather_dataroot, training_time, use_augment=False)

    dataloader_test = create_loader(
        test_set,
        batch_size=1,
        shuffle=False,
        is_training=False,
        pin_memory=False,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher)

    return dataloader_test


def load_dataset(data_dir):
    ds = []
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    pt_files = sorted(pt_files)
    for i in range(len(pt_files)):
        ds.append(torch.load(pt_files[i]))
    return ds


class PrefetchLoader:

    def __init__(self, loader, mean=None, std=None, channels=3, fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean
                                      ]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std
                                     ]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(
                            self.std)
                        next_target = next_target.half().sub_(self.mean).div_(
                            self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(
                            self.std)
                        next_target = next_target.float().sub_(self.mean).div_(
                            self.std)
                    else:
                        next_input = next_input.float()
                        next_target = next_target.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class gnconv(nn.Module):

    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], 1)
            for i in range(order - 1)
        ])

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims,
              'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class HorBlock(nn.Module):
    """ HorNet block """

    def __init__(self,
                 dim,
                 order=4,
                 mlp_ratio=4,
                 drop_path=0.,
                 init_value=1e-6,
                 gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim, order)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(
            mlp_ratio *
            dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma1 = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        gamma1 = self.gamma1.view(C, 1, 1)
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 init_value=1e-5,
                 act_layer=nn.GELU,
                 norm_layer=GroupNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


from timm.models.layers import DropPath, trunc_normal_


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class MixMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(
            hidden_features)  # CFF: Convlutional feed-forward network
        self.act = act_layer()  # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):

    def __init__(self, d_model, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class VANBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 init_value=1e-2,
                 act_layer=nn.GELU,
                 attn_shortcut=True):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, attn_shortcut=attn_shortcut)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
            act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(
                GroupConv2d(
                    C_hid,
                    C_out,
                    kernel_size=ker,
                    stride=1,
                    padding=ker // 2,
                    groups=groups,
                    act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            dd_k,
            stride=1,
            padding=dd_p,
            groups=dim,
            dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution

        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP"""

    def __init__(self,
                 dim,
                 kernel_size=21,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.1,
                 init_value=1e-2,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


class ConvMixerSubBlock(nn.Module):
    """A block of ConvMixer."""

    def __init__(self, dim, kernel_size=9, activation=nn.GELU):
        super().__init__()
        # spatial mixing
        self.conv_dw = nn.Conv2d(
            dim, dim, kernel_size, groups=dim, padding="same")
        self.act_1 = activation()
        self.norm_1 = nn.BatchNorm2d(dim)
        # channel mixing
        self.conv_pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.act_2 = activation()
        self.norm_2 = nn.BatchNorm2d(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        x = x + self.norm_1(self.act_1(self.conv_dw(x)))
        x = self.norm_2(self.act_2(self.conv_pw(x)))
        return x


class ConvNeXtSubBlock(ConvNeXtBlock):
    """A block of ConvNeXt."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(
            dim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            ls_init_value=1e-6,
            conv_mlp=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma'}

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma.reshape(1, -1, 1, 1) *
            self.mlp(self.norm(self.conv_dw(x))))
        return x


class HorNetSubBlock(HorBlock):
    """A block of HorNet."""

    def __init__(self, dim, mlp_ratio=4., drop_path=0.1, init_value=1e-6):
        super().__init__(
            dim,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            init_value=init_value)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma1', 'gamma2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MLPMixerSubBlock(MixerBlock):
    """A block of MLP-Mixer."""

    def __init__(self,
                 dim,
                 input_resolution=None,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.1):
        seq_len = input_resolution[0] * input_resolution[1]
        super().__init__(
            dim,
            seq_len=seq_len,
            mlp_ratio=(0.5, mlp_ratio),
            drop_path=drop_path,
            drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(
            self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MogaSubBlock(nn.Module):
    """A block of MogaNet."""

    def __init__(self,
                 embed_dims,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4]):
        super(MogaSubBlock, self).__init__()
        self.out_channels = embed_dims
        # spatial attention
        self.norm1 = nn.BatchNorm2d(embed_dims)
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # channel MLP
        self.norm2 = nn.BatchNorm2d(embed_dims)
        mlp_hidden_dims = int(embed_dims * mlp_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims,
            mlp_hidden_dims=mlp_hidden_dims,
            ffn_drop=drop_rate)
        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2', 'sigma'}

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class PoolFormerSubBlock(PoolFormerBlock):
    """A block of PoolFormer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(
            dim,
            pool_size=3,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop=drop,
            init_value=1e-5)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self,
                 dim,
                 input_resolution=None,
                 layer_i=0,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(
            4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(
            dim,
            input_resolution,
            num_heads=8,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop=drop,
            qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class CBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_value=1e-6,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        # layer scale
        self.gamma_1 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_1', 'gamma_2'}

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class VANSubBlock(VANBlock):
    """A block of VAN."""

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 init_value=1e-2,
                 act_layer=nn.GELU):
        super().__init__(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value,
            act_layer=act_layer)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class ViTSubBlock(ViTBlock):
    """A block of Vision Transformer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(
            dim=dim,
            num_heads=8,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop=drop,
            drop_path=drop_path,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = TemporalAttentionModule(
            d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            dd_k,
            stride=1,
            padding=dd_p,
            groups=dim,
            dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),  # expansion
            nn.Sigmoid())

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution
        f_x = self.conv1(attn)  # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u


class TAUSubBlock(GASubBlock):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self,
                 dim,
                 kernel_size=21,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.1,
                 init_value=1e-2,
                 act_layer=nn.GELU):
        super().__init__(
            dim=dim,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value,
            act_layer=act_layer)

        self.attn = TemporalAttention(dim, kernel_size)


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(
                C_in,
                C_hid,
                spatio_kernel,
                downsampling=samplings[0],
                act_inplace=act_inplace), *[
                    ConvSC(
                        C_hid,
                        C_hid,
                        spatio_kernel,
                        downsampling=s,
                        act_inplace=act_inplace) for s in samplings[1:]
                ])

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[
                ConvSC(
                    C_hid,
                    C_hid,
                    spatio_kernel,
                    upsampling=s,
                    act_inplace=act_inplace) for s in samplings[:-1]
            ],
            ConvSC(
                C_hid,
                C_hid,
                spatio_kernel,
                upsampling=samplings[-1],
                act_inplace=act_inplace))
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self,
                 channel_in,
                 channel_hid,
                 N2,
                 incep_ker=[3, 5, 7, 11],
                 groups=8,
                 **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [
            gInception_ST(
                channel_in,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups)
        ]
        for i in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups))
        enc_layers.append(
            gInception_ST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups))
        dec_layers = [
            gInception_ST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                incep_ker=incep_ker,
                groups=groups)
        ]
        for i in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups))
        dec_layers.append(
            gInception_ST(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                incep_ker=incep_ker,
                groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 input_resolution=None,
                 model_type=None,
                 mlp_ratio=8.,
                 drop=0.0,
                 drop_path=0.0,
                 layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels,
                kernel_size=21,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(
                in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels,
                input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop_rate=drop,
                drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels,
                input_resolution,
                layer_i=layer_i,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels,
                kernel_size=21,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(
            z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self,
                 channel_in,
                 channel_hid,
                 N2,
                 input_resolution=None,
                 model_type=None,
                 mlp_ratio=4.,
                 drop=0.0,
                 drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)
        ]

        # downsample
        enc_layers = [
            MetaBlock(
                channel_in,
                channel_hid,
                input_resolution,
                model_type,
                mlp_ratio,
                drop,
                drop_path=dpr[0],
                layer_i=0)
        ]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(
                MetaBlock(
                    channel_hid,
                    channel_hid,
                    input_resolution,
                    model_type,
                    mlp_ratio,
                    drop,
                    drop_path=dpr[i],
                    layer_i=i))
        # upsample
        enc_layers.append(
            MetaBlock(
                channel_hid,
                channel_in,
                input_resolution,
                model_type,
                mlp_ratio,
                drop,
                drop_path=drop_path,
                layer_i=N2 - 1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


class Config:

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')

        if filename is not None:
            cfg_dict = self._file2dict(filename, True)
            filename = filename

        super(Config, self).__setattr__('_cfg_dict', cfg_dict)
        super(Config, self).__setattr__('_filename', filename)

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, 'r') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py']:
            raise IOError('Only py type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            temp_config_name = osp.basename(temp_config_file.name)

            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
            # close temp file
            temp_config_file.close()
        return cfg_dict

    @staticmethod
    def fromfile(filename, use_predefined_variables=True):
        cfg_dict = Config._file2dict(filename, use_predefined_variables)
        return Config(cfg_dict, filename=filename)


class Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self,
                 in_shape,
                 hid_S=16,
                 hid_T=256,
                 N_S=4,
                 N_T=4,
                 model_type='gSTA',
                 mlp_ratio=8.,
                 drop=0.0,
                 drop_path=0.0,
                 spatio_kernel_enc=3,
                 spatio_kernel_dec=3,
                 act_inplace=True,
                 **kwargs):
        super(Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        C_out = 50
        H, W = int(H / 2**(N_S / 2)), int(
            W / 2**(N_S / 2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(
            C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(
            hid_S, C_out, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        if model_type == 'incepu':
            self.hid = MidIncepNet(T * hid_S, hid_T, N_T)
        else:
            self.hid = MidMetaNet(
                T * hid_S,
                hid_T,
                N_T,
                input_resolution=(H, W),
                model_type=model_type,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        C_out = 5
        # print(Y.shape)
        Y = Y.reshape(B, 10 * T, C_out, H, W)

        return Y


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str): Name of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        level (int): Used level in the multi-variant version.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self,
                 data_root,
                 training_time,
                 transform_data=None,
                 transform_labels=None,
                 use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data = None
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment
        self.ds = load_dataset(self.data_root)
        self.num_step = 20  # for 5-days
        self.num_data = len(self.ds)
        self.valid_idx = self.num_data

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(
            seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x + h, y:y + w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx

    def __getitem__(self, index):

        input = self.ds[index]  # you can use subset of input, eg: only surface
        input_x = np.zeros(
            (input.shape[0], 10, input.shape[2], input.shape[3]))
        atom_channels = [12, 25, 38, 51, 64, 65, 66, 67, 68, 69]
        # atom_channels = [0,6,12,13,19,25,26,32,38,39,45,51,52,58,64,65,66,67,68,69]
        # atom_channels = [65,66,67,68,69]
        for i in range(len(atom_channels)):

            input_x[:, i, :, :] = input[:, atom_channels[i], :, :]

        input_ = np.zeros((input_x.shape[0], input_x.shape[1], 128, 128))
        vectorized_resize = np.vectorize(resize_func, signature='(n,m)->(p,q)')
        input_ = vectorized_resize(input_x)
        input = torch.from_numpy(input_)
        input = torch.nan_to_num(input)  # t c h w

        return input


def infer(data_loader, length=None):
    """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
    # preparation
    results = []

    length = len(data_loader.dataset) if length is None else length

    # loop
    for idx, batch_x in enumerate(data_loader):
        with torch.no_grad():
            batch_x = batch_x.to(device)
            batch_x = batch_x.float()

            pred_y = model(batch_x)
            results.append(pred_y.cpu().numpy())

        if args.empty_cache:
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    best_model_path = 'checkpoint099.pth'
    data_dir = '../data/test/input/'
    save_dir = '../submit/output/'

    args = create_parser().parse_args()
    config = args.__dict__
    dataset_parameters = {
        'era5': {
            'in_shape': [2, 10, 128, 128],
            'pre_seq_length': 2,
            'aft_seq_length': 20,
            'total_length': 22,
            'data_name': 'era5',
            'metrics': ['mse', 'mae', 'ssim', 'psnr']
        }
    }

    assert args.config_file is not None, "Config file is required for testing"
    config = update_config(
        config,
        load_config(args.config_file),
        exclude_keys=['method', 'batch_size', 'val_batch_size'])
    config['test'] = True
    config.update(dataset_parameters['era5'])

    device = torch.device('cuda:0')
    model = Model(**config).to(device)
    model.load_state_dict(torch.load(best_model_path))

    dataloader_test = load_data(
        batch_size=1,
        val_batch_size=1,
        data_root=data_dir,
        training_time=None,
        num_workers=1,
        use_augment=False)

    results = infer(dataloader_test)

    for i in range(len(results)):
        pred = results[i][0]

        pred_ = np.zeros((pred.shape[0], 1, 161, 161))
        vectorized_resize = np.vectorize(
            resize_back_func, signature='(n,m)->(p,q)')
        pred_ = vectorized_resize(pred)

        pred_pt = torch.from_numpy(pred_)
        pred_pt = pred_pt.to(torch.float16)
        torch.save(pred_pt, os.path.join(save_dir, str(i).zfill(3) + ".pt"))

    print("end")