# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/gldas_demo/')
from openstl.api import BaseExperiment
from openstl.utils import (create_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    # events = ['0ll8.tif','4ay5.tif','5gu2.tif']


    # for event in events:
    args = create_parser().parse_args()
    config = args.__dict__
    # print(event)
    # config['event'] = event
    # config['resume_from'] = "/home/gldas_demo/work_dirs/gldas_tau_sub/checkpoints/latest.pth"

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                            exclude_keys=['method'])
    else:
        config = update_config(config, load_config(cfg_path),
                            exclude_keys=['method', 'batch_size', 'val_batch_size', 'sched',
                                            'drop_path', 'warmup_epoch'])

    # set multi-process settings
    setup_multi_processes(config)

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)
