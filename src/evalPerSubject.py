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
from matplotlib.colors import ListedColormap

# remove dot if debug

from .eval_utils.eval_process import eval_preprocess, eval_postprocess, getImages
from .eval_utils.evaluation_metrics import eval_metrics
from .datasets.dataset import GE3T_preprocessing, Utrecht_preprocessing


def load_test_imgs(heldout):
    print(f'Loading test domain: {heldout} ...')
    test_path = f'path/{heldout}/test.txt'
    lines = [p.strip().split() for p in open(test_path, 'r')]
    paths = [f'{path}' for path, _ in lines]
    return paths


def evaluatePerSubject(model,
                       heldout,
                       wandb=None,
                       T1=True,
                       savePredictions=True,
                       save_dir=None):
    paths = load_test_imgs(heldout)
    running_dsc = 0
    running_h95 = 0
    running_avd = 0
    running_recall = 0
    running_f1 = 0
    i = 0
    if heldout != 'GE3T':
        preprocess = Utrecht_preprocessing
    else:
        preprocess = GE3T_preprocessing
    for img_subj_pth in paths:
        subj_name = img_subj_pth.split('/')[2]
        img = sitk.ReadImage(img_subj_pth + '/pre/FLAIR.nii.gz')
        img = sitk.GetArrayFromImage(img)
        T1_img = None
        if T1:
            T1_img = sitk.ReadImage(img_subj_pth + '/pre/T1.nii.gz')
            T1_img = sitk.GetArrayFromImage(T1_img)
        try:
            groundTruth = sitk.ReadImage(img_subj_pth + '/wmh.nii.gz')
        except:
            groundTruth = sitk.ReadImage(img_subj_pth + '/wmh.nii')
        groundTruth = sitk.GetArrayFromImage(groundTruth)
        if heldout == 'Local':
            img = img[(24, 25, 26, 27, 28), :, :]
            if T1:
                T1_img = T1_img[(24, 25, 26, 27, 28), :, :]
            groundTruth = groundTruth[(24, 25, 26, 27, 28), :, :]

            groundTruth = np.float32(groundTruth)
        img, groundTruth, _, T1_img = preprocess(img,
                                                 groundTruth,
                                                 T1=T1_img,
                                                 train=False)
        imgOrig = img.copy()
        img = np.expand_dims(img, axis=1)
        if T1:
            T1_img = np.expand_dims(T1_img, axis=1)
            img = np.concatenate((img, T1_img), axis=1)
        img_tensor = torch.tensor(img).float().cuda()
        with torch.no_grad():
            model.eval()
            yhat, _, _ = model(img_tensor)
        yhat = torch.squeeze(yhat, dim=1)
        yhat = yhat.cpu().numpy()
        # yhat_postprocess = eval_postprocess(imgOrig, yhat)
        yhat_sitkImg = sitk.GetImageFromArray(yhat)
        groundTruth_sitkImg = sitk.GetImageFromArray(groundTruth)
        result, threshold_groundTruth, threshold_pred = eval_metrics(
            groundTruth_sitkImg, yhat_sitkImg)
        running_dsc += result['dsc']
        running_h95 += result['h95']
        running_avd += result['avd']
        running_recall += result['recall']
        running_f1 += result['f1']
        threshold_groundTruth = sitk.GetArrayFromImage(threshold_groundTruth)
        threshold_pred = sitk.GetArrayFromImage(threshold_pred)
        if savePredictions:
            plot_imgs(imgOrig, threshold_pred, threshold_groundTruth, save_dir,
                      subj_name)
        i += 1

    dsc = running_dsc / i
    h95 = running_h95 / i
    avd = running_avd / i
    recall = running_recall / i
    f1 = running_f1 / i
    if wandb is not None:
        wandb.log({
            f"{heldout}_per_subject_dsc": dsc,
            f"{heldout}_per_subject_h95": h95,
            f"{heldout}_per_subject_avd": avd,
            f"{heldout}_per_subject_recall": recall,
            f"{heldout}_per_subject_f1": f1
        })
    with open(f'{save_dir}/per_subject_results.txt', 'a') as res:
        res.write(
            f'\n{heldout}:\n dsc: {dsc} | h95: {h95} | avd: {avd} | recall: {recall} | f1: {f1}'
        )


def plot_imgs(flair, pred, groundTruth, save_dir, subj_name):
    for i in range(flair.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        ax.flat[0].imshow(flair[i, ...], cmap='gray', vmin=None, vmax=None)
        ax.flat[0].set_title('Flair')
        ax.flat[1].imshow(pred[i, ...], vmin=None, vmax=None)
        ax.flat[1].set_title('Prediction')
        ax.flat[2].imshow(groundTruth[i, ...], vmin=None, vmax=None)
        ax.flat[2].set_title('Ground Truth')
        plt.axis('off')
        plt.tight_layout()
        Path(f'{save_dir}/pred/{subj_name}').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{save_dir}/pred/{subj_name}/slice-{i}')
        plt.close(fig)

        plt.imshow(flair[i, ...],
                   cmap='gray',
                   label='flair',
                   vmin=None,
                   vmax=None)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f'{save_dir}/pred/{subj_name}/slice-{i}-flair.pdf',
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        fig, ax = plt.subplots()
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['savefig.facecolor'] = 'black'

        masked_groundtruth = np.ma.masked_where(groundTruth[i, ...] < 0.9,
                                                groundTruth[i, ...])
        masked_pred = np.ma.masked_where(pred[i, ...] < 0.9, pred[i, ...])
        # colours = ListedColormap(['#D1040D'])

        falsePositive = groundTruth[i, ...] + pred[i, ...]
        falsePositive[falsePositive == 2] = 1
        falsePositive = falsePositive - groundTruth[i, ...]
        masked_falsePositive = np.ma.masked_where(falsePositive < 0.9,
                                                  falsePositive)

        colours = ListedColormap(['red'])
        plt.imshow(masked_groundtruth,
                   cmap=colours,
                   label='ground truth',
                   vmin=0,
                   vmax=None)
        # colours = ListedColormap(['#044ED1'])
        colours = ListedColormap(['green'])
        plt.imshow(masked_pred,
                   cmap=colours,
                   label='pred',
                   vmin=0,
                   vmax=None,
                   alpha=1)
        colours = ListedColormap(['#F1D806'])
        plt.imshow(masked_falsePositive,
                   cmap=colours,
                   label='pred',
                   vmin=0,
                   vmax=None,
                   alpha=1)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(f'{save_dir}/pred/{subj_name}/slice-{i}-overlay.pdf',
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)
