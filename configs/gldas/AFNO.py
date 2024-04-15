method = 'AFNONet'
# model
patch_size = 4
in_chans = 7
out_chans = 7
num_blocks = 4
embed_dim = 20*10
# model_type = None
# training
lr = 1e-4
batch_size = 8
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0