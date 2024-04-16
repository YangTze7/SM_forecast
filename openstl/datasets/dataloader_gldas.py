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

d2r = np.pi / 180
from glob import glob
import rasterio

# Open the GeoTIFF file
def read_tiff(tiff_path):
    with rasterio.open(tiff_path) as src:
        # Access metadata
        # print(src.meta)

        # Read raster data as a NumPy array
        data = src.read(1)  # Change the band number as needed

        # Access spatial information
        # print(src.crs)
        # print(src.bounds)
        # print(src.transform)
    return data


def resize_func(x):
    return cv2.resize(x, (128, 128), interpolation=cv2.INTER_LINEAR)

def chunk_time(ds):
    dims = {k:v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def load_dataset(data_dir,start_year,end_year):
    ds = []
    if start_year==end_year:
        data_name = os.path.join(data_dir, f'weather_round1_train_{start_year}')
        x = xr.open_zarr(data_name, consolidated=True)
        ds = chunk_time(x)
    else:
        for y in range(start_year, end_year):
            data_name = os.path.join(data_dir, f'weather_round1_train_{y}')
            x = xr.open_zarr(data_name, consolidated=True)
            # print(f'{data_name}, {x.time.values[0]} ~ {x.time.values[-1]}')
            ds.append(x)
        ds = xr.concat(ds, 'time')
        ds = chunk_time(ds)
    return ds


def load_tiff_path(data_dir,var_name,period):

    paths = glob(os.path.join(data_dir,var_name,'*.tif'))
    paths = sorted(paths)
    paths = paths[period[0]:period[1]]
    return paths

def get_other_vartpath(soilm_path):

    soilt_path = soilm_path.replace('SoilMoi0_10cm_inst','SoilTMP0_10cm_inst')
    rainf_path = soilm_path.replace('SoilMoi0_10cm_inst','Rainf_f_tavg')
    lw_path = soilm_path.replace('SoilMoi0_10cm_inst','LWdown_f_tavg')
    tair_path = soilm_path.replace('SoilMoi0_10cm_inst','Tair_f_inst')
    sw_path = soilm_path.replace('SoilMoi0_10cm_inst','SWdown_f_tavg')
    wind_path = soilm_path.replace('SoilMoi0_10cm_inst','Wind_f_inst')

    path_vars = [soilm_path,soilt_path,rainf_path,lw_path,tair_path,sw_path,wind_path]

    return path_vars
    
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

    def __init__(self, data_root,training_time,step=1,x_file = "X.npy",y_file="y.npy", level=1,
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


        self.ds_path = load_tiff_path(self.data_root,'SoilMoi0_10cm_inst',self.training_time)
        self.ds_path = sorted(self.ds_path)

        self.valid_idx = self.ds_path.__len__()-20


   
        # self.X = self.X.astype(np.float16)
        # self.y = self.y.astype(np.float16)
        # np.save("X_20_r2.npy",self.X)
        # np.save("y_20_r2.npy",self.y)

        print("saved")
    def _augment_seq(self, pre_seqs,after_seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = pre_seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        pre_seqs = F.interpolate(pre_seqs, scale_factor=1 / crop_scale, mode='bilinear')
        after_seqs = F.interpolate(after_seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = pre_seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        pre_seqs = pre_seqs[:, :, x:x+h, y:y+w]
        after_seqs = after_seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            pre_seqs = torch.flip(pre_seqs, dims=(3, ))  # horizontal flip
            after_seqs = torch.flip(after_seqs, dims=(3, ))  # horizontal flip
        return pre_seqs,after_seqs

    def __len__(self):
        return self.valid_idx

    def __getitem__(self, index):
        
        
        
        var_data = []
        for t in range(12):
            idx = index+t
            soilm_path = self.ds_path[idx]
            path_vars = get_other_vartpath(soilm_path)
            var_data_step = [read_tiff(p) for p in path_vars]
            var_data.append(var_data_step)
        var_data = np.array(var_data)

        


        input = var_data[0:0+8]
        target = var_data[8:12]
  





        # input_ = np.zeros((input.values.shape[0], input.values.shape[1], 128, 128))
        input_ = np.zeros((input.shape[0], input.shape[1], 128, 256))
        vectorized_resize = np.vectorize(resize_func, signature='(n,m)->(p,q)')
        input_ = vectorized_resize(input)
        
        # input_ = np.repeat(input_, repeats=10, axis=0)

        # target_ = np.zeros((target.values.shape[0], target.values.shape[1], 128, 128))
        target_ = np.zeros((target.shape[0], target.shape[1], 128, 256))
        vectorized_resize = np.vectorize(resize_func, signature='(n,m)->(p,q)')
        target_ = vectorized_resize(target)

        input = torch.from_numpy(input_)
        target = torch.from_numpy(target_)
        
        input = torch.nan_to_num(input) # t c h w 
        target = torch.nan_to_num(target) # t c h w 

        # tx2 = time.perf_counter()
        # print(tx2-tx1)

        if self.use_augment:

            input,target = self._augment_seq(input,target, crop_scale=0.96)


        return input, target


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              train_time=[0,5000],
              val_time=[5000,7000],
              test_time=[5000,7000],
              step=1,
              level=1,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    weather_dataroot = data_root 

    train_set = WeatherBenchDataset(data_root=weather_dataroot,
                                    training_time=train_time,x_file="X_20_r1.npy",y_file="y_20_r1.npy",
                                    step=step, level=level, use_augment=use_augment)

    vali_set = WeatherBenchDataset(weather_dataroot,
                                    training_time=val_time,x_file="X_20_val.npy",y_file="y_20_val.npy",
                                    step=step, level=level, use_augment=False)

    test_set = WeatherBenchDataset(weather_dataroot,
                                    training_time=test_time,x_file="X_20_val.npy",y_file="y_20_val.npy",
                                    step=step, level=level, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=False, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)

    dataloader_vali = create_loader(test_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=False, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=False, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    from openstl.core import metric

    # data_split=['5_625',]
    # data_name = 'mv'
    
    data_dir = '/home/gldas/gldas_vars_clipped' # change to you dataset dir
    step  = 1

    level = 1

    dataloader_train, _, dataloader_test = \
        load_data(batch_size=32,
                val_batch_size=32,
                data_root=data_dir,
                num_workers=4, 
                train_time=[0,5000],
                val_time=[5000,7000],
                test_time=[5000,7000],
                step=step, level=level, use_augment=True)

    print(len(dataloader_train), len(dataloader_test))
    # x,y = dataloader_train[0]
    for item in dataloader_train:
        print('train', item[0].shape)
        
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