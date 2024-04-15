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
import random
from functools import partial
from itertools import repeat
from typing import Callable
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler

import torch.utils.data
import numpy as np

from glob import glob
def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def fast_collate_for_prediction(batch):
    """ A fast collation function optimized for float32 images (np array or torch)
        and float32 targets (video prediction labels) in video prediction tasks"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into
        # one tensor ordered by position such that all tuple of position n will end up
        # in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.float32)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            # all input tensor tuples must be same length
            assert len(batch[i][0]) == inner_tuple_size
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.zeros((batch_size, *batch[1][0].shape), dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)        
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
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
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
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
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


def reshape_patch(img_tensor, patch_size):
    assert 4 == img_tensor.ndim
    seq_length = np.shape(img_tensor)[0]
    img_height = np.shape(img_tensor)[1]
    img_width = np.shape(img_tensor)[2]
    num_channels = np.shape(img_tensor)[3]
    a = np.reshape(img_tensor, [seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0, 1, 3, 2, 4, 5])
    patch_tensor = np.reshape(b, [seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size * patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor


def reshape_patch_back_tensor(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    patch_narray = patch_tensor.detach().cpu().numpy()
    batch_size = np.shape(patch_narray)[0]
    seq_length = np.shape(patch_narray)[1]
    patch_height = np.shape(patch_narray)[2]
    patch_width = np.shape(patch_narray)[3]
    channels = np.shape(patch_narray)[4]
    img_channels = channels // (patch_size * patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                     patch_height, patch_width,
                                     patch_size, patch_size,
                                     img_channels])
    b = a.permute([0, 1, 2, 4, 3, 5, 6])
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                   patch_height * patch_size,
                                   patch_width * patch_size,
                                   img_channels])
    return img_tensor.permute(0, 1, 4, 2, 3)


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
        self.label = None
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
            range(-idx_in[0], self.series_path.__len__()-idx_in[-1]-1))

    

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

        index = self.valid_idx[index]
        data_ts = []
        for i in range(index+self.idx_in.min(),index+self.idx_in.max()+1):
            if i == int(len(self.idx_in)/2):
                self.label = read_tif_as_numpy(self.label_files[i])
                continue
            label_array = read_tif_as_numpy(self.label_files[i])
            PS_array = read_tif_as_numpy(self.PS_files[i])
            S1_array = read_tif_as_numpy(self.S1_files[i])
            tmp_array = np.stack((PS_array,label_array),axis=0)

            data_ts.append(tmp_array)
        self.data = np.array(data_ts,dtype=np.float32) 
        

        # index = self.valid_idx[index]
        data = torch.tensor(self.data)
        labels = torch.tensor(self.label)


        # index = self.valid_idx[index]
        # data = torch.tensor(self.data[0:len(self.idx_in)])
        # labels = torch.tensor(self.data[len(self.idx_in):len(self.idx_in)+len(self.idx_out)])

        # if self.use_augment:
        #     len_data = self.idx_in.shape[0]
        #     seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
        #     data = seqs[:len_data, ...]
        #     labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size=8, val_batch_size=8,data_root="/home/convlstm_predict/OpenSTL-master/data/SCJ_exp/test",num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False,use_prefetcher=False, drop_last=False,
              idx_in=[-1,0,1],
              idx_out=0,
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
                idx_in=[-1,0,1],
                idx_out=0,
                step=step, use_augment=True)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
    
    print("end")