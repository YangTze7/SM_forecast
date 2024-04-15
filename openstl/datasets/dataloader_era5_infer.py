# %%
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
from openstl.datasets.utils import create_loader

import cv2
try:
    import xarray as xr
except ImportError:
    xr = None

import glob
d2r = np.pi / 180



def resize_func(x):
    return cv2.resize(x, (128, 128), interpolation=cv2.INTER_LINEAR)

def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def load_dataset(data_dir,start_year,end_year):
    ds = []
    pt_files = glob.glob(os.path.join(data_dir,"*.pt"))
    pt_files = sorted(pt_files)
    for i in range(len(pt_files)):
        ds.append(torch.load(pt_files[i]))
    return ds




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

    def __init__(self, data_root,training_time,step=1, level=1,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.training_time = training_time
        self.step = step
        self.level = level
        self.data = None

        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment


        self.ds = load_dataset(self.data_root,training_time[0],training_time[-1])

        self.num_step = 20 # for 5-days

        self.num_data = len(self.ds)


        self.valid_idx = self.num_data

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx

    def __getitem__(self, index):
        

        
        input = self.ds[index] # you can use subset of input, eg: only surface 
        input_x = np.zeros((input.shape[0],10,input.shape[2],input.shape[3]))
        atom_channels = [12,25,38,51,64,65,66,67,68,69]
        # atom_channels = [0,6,12,13,19,25,26,32,38,39,45,51,52,58,64,65,66,67,68,69]
        # atom_channels = [65,66,67,68,69]
        for i in range(len(atom_channels)):

            input_x[:,i,:,:]=input[:,atom_channels[i],:,:]

        input_ = np.zeros((input_x.shape[0],input_x.shape[1],128, 128))
        vectorized_resize = np.vectorize(resize_func, signature='(n,m)->(p,q)')
        input_ = vectorized_resize(input_x)

        # target_ = np.zeros((target.values.shape[0], target.values.shape[1], 128, 128))
        # vectorized_resize = np.vectorize(resize_func, signature='(n,m)->(p,q)')
        # target_ = vectorized_resize(target.values)
        # input_ = input_[:,5,:,:]
        # input_ = input_[:,np.newaxis,:,:]
        input = torch.from_numpy(input_)
        # target = torch.from_numpy(target_)
        
        input = torch.nan_to_num(input) # t c h w 
        # target = torch.nan_to_num(target) # t c h w 
        return input


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              train_time=[2007],
              val_time=[2010],
              test_time=[2011],
              step=1,
              level=1,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    weather_dataroot = data_root 



    test_set = WeatherBenchDataset(weather_dataroot,
                                    training_time=test_time,
                                    step=step, level=level, use_augment=False)


    dataloader_test = create_loader(test_set,
                                    batch_size=1,
                                    shuffle=False, is_training=False,
                                    pin_memory=False, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_test


if __name__ == '__main__':
    from openstl.core import metric

    # data_split=['5_625',]
    # data_name = 'mv'

    data_dir = '/home/convlstm_predict/2023tianchi_weather_prediction/weather_round1_test/input' # change to you dataset dir
    step  = 1

    level = 1

    dataloader_test = \
        load_data(batch_size=32,
                val_batch_size=32,
                data_root=data_dir,
                num_workers=4, 
                train_time=[2007,2008,2009,2010],
                val_time=[2011],
                test_time=[2011],
                step=step, level=level, use_augment=False)

    print(len(dataloader_test))
    # x,y = dataloader_train[0]

        
    for item in dataloader_test:
        print('test', item[0].shape)
        break








    # daily climate baseline
    climates = {
        't2m': 3.1084048748016357,
        'u10': 4.114771819114685,
        'v10': 4.184110546112061,
        'msl': 729.5839385986328,
        'tp': 0.49046186606089276,
    }


    def compute_rmse(out, tgt):
        rmse = torch.sqrt(((out - tgt)**2).mean())
        return rmse


    def run_eval(output, target):
        '''
            result: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
            target: (batch x step x channel x lat x lon), eg: N x 20 x 5 x H x W
        '''
        result = {}
        for cid, (name, clim) in enumerate(climates.items()):
            res = []
            for sid in range(output.shape[1]):
                out = output[:, sid, cid]
                tgt = target[:, sid, cid]
                rmse = compute_rmse(out, tgt)
                nrmse = (rmse - clim) / clim
                res.append(nrmse)
                
                # normalized rmse, lower is better,
                # 0 means equal to climate baseline, 
                # less than 0 means better than climate baseline,   
                # -1 means perfect prediction            

            score = max(0, -np.mean(res))
            result[name] = float(score)

        score = np.mean(list(result.values()))
        result['score'] = float(score) 
        return result



    # run eval on the normalized output and target, but the online evalation will un-normalized your output to origial scale

    print("end")
