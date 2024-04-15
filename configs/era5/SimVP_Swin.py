method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'swin'
hid_S = 4
hid_T = 16
N_T = 4
N_S = 4
# training
lr = 1e-3
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0