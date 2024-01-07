import os
import time
import logging
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from utils.util import setup_logger, print_args
from models import Trainer

def main():
    parser = argparse.ArgumentParser(description='referenceSR Testing')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='UNetWaveletNet_test', type=str)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='MASA', type=str, help='')
    parser.add_argument('--input_nc', default=1, type=int)
    parser.add_argument('--output_nc', default=1, type=int)

    ## dataloader setting
    parser.add_argument('--testdata_root', default='/data0/czy/M4Raw/denoising_demo/multicoil_test/',type=str)
    parser.add_argument('--dataset', default='M4Raw', type=str, help='M4Raw | fastMRI | DlDegibbs | ARMNet')
    parser.add_argument('--modal', default='T1', type=str, help='T1 | T2 | FLAIR | MutilContrast')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_augmentation', default=False, type=bool)
    
    parser.add_argument('--resume', default='./pretrained_weights/masa_rec.pth', type=str)
    parser.add_argument('--testset', default='TestSet', type=str, help='Sun80 | Urban100 | TestSet_multi')
    parser.add_argument('--save_folder', default='./test_results/', type=str)

    ## UnetModel_arch：
    parser.add_argument('--in_chans', default=1, type=int)
    parser.add_argument('--out_chans', default=1, type=int)
    parser.add_argument('--chans', default=256, type=int)
    parser.add_argument('--num_pool_layers', default=4, type=int)
    parser.add_argument('--drop_prob', default=0.0, type=float)
    
    ## UNetWavelet:
    parser.add_argument('--nlayers', default=10, type=int)
    
    ## setup training environment
    args = parser.parse_args()

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        
    ## The current directory folder is designated as 'test_results' for saving files
    args.test_save_dir = os.path.join(args.save_folder, 'test1')
    args.save_folder = os.path.join(args.save_folder, args.testset, args.name)
    
    ## Create the testing image folder if it does not exist
    if os.path.exists(args.test_save_dir) == False:
        os.makedirs(args.test_save_dir)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    setup_logger(log_file_path)

    print_args(args)
    cudnn.benchmark = True

    ## test model
    trainer = Trainer(args)
    trainer.test()

if __name__ == '__main__':
    main()
