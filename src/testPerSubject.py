from pathlib import Path
import argparse
import sys
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random
import SimpleITK as sitk
import torchvision.transforms.functional as augmentor

# remove dot if debug
from .datasets.utils import get_splits, postprocessing
from .models.models import UNet, LightWeight
from .models.heads import UNetDiscriminator
from .models.discriminator import Discriminator
from .models.losses import HLoss, DSCLoss, DSCLoss_2
from .datasets.dataset import Augmentation
from .models.metrics import Critic
from .models.utils import parser_add_argument, validation_eval, test_eval, set_random_seed
from .models.adv_exp import train_adversarial_examples
from .eval_utils.eval_process import eval_preprocess, eval_postprocess, getImages
from .evalPerSubject import evaluatePerSubject

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='GPU idx to run', default='1,2')
args = parser.parse_args()

results_path = 'results_prev_not_local/'
results_paths = os.listdir(results_path)
results_paths.sort()
for rp in results_paths:
    model_dir = rp
    domain = os.listdir(results_path + model_dir)
    domain = domain[0]
    path_dir = results_path + model_dir + "/" + domain
    trainedModelDir = path_dir + '/last_epoch_model.pt'

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        gpus = [int(x.strip()) for x in args.gpu.split(',')]

    model = UNet(T1=True)
    if use_cuda:
        gpu_ids = []
        for i in range(len(gpus)):
            gpu_ids.append(i)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(trainedModelDir))
    save_dir = 'results_per_subject/' + model_dir + '/' + domain
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    evaluatePerSubject(model, domain, wandb=None, T1=True, save_dir=save_dir)
