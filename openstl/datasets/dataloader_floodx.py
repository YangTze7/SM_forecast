import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import rasterio
import cv2
from openstl.datasets.utils import create_loader

def read_tif_as_numpy(tif_file):
    # with rasterio.open(tif_file) as dataset:
    #     tif_array = dataset.read()
    tif_array = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
    return tif_array


class FloodSTDataset(Dataset):
    """Single Band spatial-temporal Dataset

    Args:
        data_root (str): Path to the dataset.
       
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.

        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, split,event,
                 idx_in, idx_out, step=1, 
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.event = event
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.data = None
        self.mean = None
        self.std = None
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment


        self.time = None
        self.path = None

        # get all tif file path from data_root
        series_dir= "series"
        self.path = []
        dataset_root = osp.join(self.data_root,series_dir)

        self.series_path = os.listdir(dataset_root)
        self.series_path.sort()

        events = []

        for root, dirs, files in os.walk(osp.join(dataset_root,self.series_path[0],"labels")):
            for file in files:
                if file.endswith(".tif"):
                    events.append(os.path.join(root, file))

        self.label_files = [osp.join(dataset_root,sp,"labels",self.event) for sp in self.series_path]
        self.PS_files = [osp.join(dataset_root,sp,"PS",self.event) for sp in self.series_path]
        self.S1_files = [osp.join(dataset_root,sp,"S1",self.event) for sp in self.series_path]






        self.valid_idx = np.array(
            range(-idx_in[0], self.series_path.__len__()-idx_out[-1]-1))

    

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
        return self.valid_idx.shape[0]

    def __getitem__(self, index):

        data_ts = []
        for i in range(index+self.idx_in.min(),index+self.idx_out.max()+1):
            label_array = read_tif_as_numpy(self.label_files[i])
            PS_array = read_tif_as_numpy(self.PS_files[i])
            # S1_array = read_tif_as_numpy(self.S1_files[i])
            tmp_array = np.stack((label_array,PS_array),axis=0)

            data_ts.append(tmp_array)
        self.data = np.array(data_ts,dtype=np.float32) 


        index = self.valid_idx[index]
        data = torch.tensor(self.data[0:len(self.idx_in)])
        labels = torch.tensor(self.data[len(self.idx_in):len(self.idx_in)+len(self.idx_out)])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size=8, val_batch_size=8,data_root="/home/convlstm_predict/OpenSTL-master/data/SCJ_exp/test",num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False,use_prefetcher=False, drop_last=False,
              idx_in=[-2,-1,0],
              idx_out=[1],
              event = None,
              step=1):
    # data_root = os.path.join(data_root, 'flood')
    batch_size = 1
    val_batch_size = 1

    train_set = FloodSTDataset(data_root=data_root,
                                    split="A",
                                    event=event,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step,use_augment=use_augment)
    test_set = FloodSTDataset(data_root=data_root,
                                    split="B",
                                    event=event,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step,use_augment=use_augment)
    xx = train_set[0]
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,)
    dataloader_vali = create_loader(test_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,
                                    )
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=True,)

    return dataloader_train, dataloader_vali, dataloader_test

   

    

    # return train_set


if __name__ == '__main__':


    step = 1
    dataloader_train, _, dataloader_test = \
                load_data(batch_size=8, val_batch_size=8,
                data_root=r'/home/convlstm_predict/OpenSTL-master/data/SCJ_exp/test',
                event="0ll8.tif",
                idx_in=[-2,-1,0],
                idx_out=[1],
                step=step, use_augment=True)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
    
    print("end")