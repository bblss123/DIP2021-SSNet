import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

from models.CSRNet import CSRNet
from models.SSNet import SSNet
from datasets.shanghaitechparta_dataloader import get_train_shanghaitechpartA_dataloader, get_test_shanghaitechpartA_dataloader
from datasets.SHHB_dataloader import *

# from nwpudataloader import create_test_nwpu_dataloader
import numpy as np
import time
import os
import sys
import errno
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test crowdcounting model')

parser.add_argument('--dataset', type=str, default='SHHB')
parser.add_argument('--test-files', type=str, default='.\\datasets\\ShanghaiTechPartB\\fullresolution\\origin\\test\\test.txt')
parser.add_argument('--best-model', type=str, default='.\\checkpoints\\bestvalmodel.pth')
parser.add_argument('--use-avai-gpus', action='store_true')
parser.add_argument('--gpu-devices', type=str, default='0')
parser.add_argument('--model', type=str, default='SSNet')
parser.add_argument('--test-batch', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

# parser.add_argument('--checkpoints', type=str, default='./checkpoints')

args = parser.parse_args()

criterion = nn.MSELoss(reduction='sum')

if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()

if use_gpu:
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
else:
    print("Currently using CPU (GPU is highly recommended)")

if args.dataset == 'SHHB':
    test_loader = get_test_SHHB_dataloader(file_list=args.test_files)
elif args.dataset == 'nwpu':
    # test_loader = create_test_nwpu_dataloader(file_list=args.test_filelist)
    pass

if args.model == 'CSRNet':
    model = CSRNet().cuda()
elif args.model == 'SSNet':
    model = SSNet().cuda()

if os.path.isfile(args.best_model):
    pkl = torch.load(args.best_model)
    state_dict = pkl['state_dict']
    # print("Currently epoch {}".format(pkl['epoch']))
    # model.load_state_dict(state_dict)
    model.load_state_dict({k.replace("module.",""):v for k,v in state_dict.items()})

if __name__ == '__main__':

    model.eval()

    if args.dataset == 'SHHB':
        with torch.no_grad():
            epoch_mae = 0.0
            epoch_rmse_loss = 0.0
            for i, data in enumerate(tqdm(test_loader)):
                image = data['image'].cuda()
                gt_densitymap = data['densitymap'].cuda()
                et_densitymap = model(image).detach()

                mae = abs(et_densitymap.data.sum() - gt_densitymap.sum())
                rmse = mae * mae

                epoch_mae += mae.item()
                epoch_rmse_loss += rmse.item()

            epoch_mae /= len(test_loader.dataset)
            epoch_rmse_loss = math.sqrt(epoch_rmse_loss / len(test_loader.dataset))
        print("bestmae: ", epoch_mae)
        print("rmse: ", epoch_rmse_loss)
