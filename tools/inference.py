# Simple script for inference
# Input: 2 images (left and right)
# Then using the DSGN2 Backbone, processes it and detects vehicles and pedestrians on the image (3D Object Detection)
# Output: 1. 2 images (left and right) with the detected objects drawn on them
# 2. Birds eye view map with the detected objects drawn on it

import argparse
import os
import re
import sys
import glob
from pathlib import Path
import numpy as np
import torch
import time
import cv2

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, update_cfg_by_args, log_config_to_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.utils.torch_utils import *
from pcdet.utils import cv2_utils

torch.backends.cudnn.benchmark = True

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    # basic testing options
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--exp_name', type=str, default='default', help='exp path for this experiment')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    # loading options
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to evaluate')
    parser.add_argument('--ckpt_id', type=int, default=None, help='checkpoint id to evaluate')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    # distributed options
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # config options
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument('--trainval', action='store_true', default=False, help='')
    parser.add_argument('--imitation', type=str, default="2d")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    update_cfg_by_args(cfg, args)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '_'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    assert args.ckpt or args.ckpt_id, "pls specify ckpt or ckpt_dir or ckpt_id"

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

class Inference():
    def __init__(model, dataloader, logger):
        self.model = model
        self.dataloader = dataloader
        self.logger = logger
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()
        model.eval()

    def inference(self, batch_dict):
        self.model.forward(batch_dict)
        

