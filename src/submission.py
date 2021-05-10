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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='GPU idx to run', default='0')
args = parser.parse_args()
inputDir = 'input/'
outputDir = 'output/'
trainedModelDir = 'trained_model/'

flairImage = sitk.ReadImage(os.path.join(inputDir, 'pre', 'FLAIR.nii.gz'))

flairImageOrig = sitk.GetArrayFromImage(flairImage)
groundTruth = sitk.ReadImage(os.path.join(inputDir, 'wmh.nii.gz'))

flairImage = eval_preprocess(flairImageOrig)
flairImage = np.expand_dims(flairImage, axis=1)
# resultImage = sitk.GetImageFromArray(flairImage)
# Path(outputDir).mkdir(parents=True, exist_ok=True)
# sitk.WriteImage(resultImage, os.path.join(outputDir, 'flair.nii.gz'))
flairImage_tensor = torch.tensor(flairImage).float()

if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    gpus = [int(x.strip()) for x in args.gpu.split(',')]

model = UNet()
if use_cuda:
    gpu_ids = []
    for i in range(len(gpus)):
        gpu_ids.append(i)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
else:
    model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load(f'{trainedModelDir}/model.pt'))
with torch.no_grad():
    model.eval()
    x = flairImage_tensor.cuda()
    yhat, _, _ = model(x)
yhat = torch.squeeze(yhat, dim=1)
yhat = yhat.cpu().numpy()
orig_pred = eval_postprocess(flairImageOrig, yhat)

resultImage = sitk.GetImageFromArray(orig_pred)
# testImage, resultImage = getImages(groundTruth, resultImage)

Path(outputDir).mkdir(parents=True, exist_ok=True)
sitk.WriteImage(resultImage, os.path.join(outputDir, 'result.nii.gz'))
