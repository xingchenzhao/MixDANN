import os
from torchvision import transforms
import torch
from .dataset import LazyLoader, ImageSet, Mixed
import numpy as np

WMH_SEG = ['Utrecht', 'GE3T', 'Singapore']
DATASETS = {'WMH_SEG': WMH_SEG}
DOMAINS = {'WMH_SEG': 2}
SPLITS = ['train', 'test', 'val']


def get_splits(name, T1=False, whitestripe=False, test_on_local=False):
    if test_on_local:
        WMH_SEG.insert(0, 'Local')
        DOMAINS['WMH_SEG'] = DOMAINS['WMH_SEG'] + 1
    return {
        heldout: {
            'train':
            LazyLoader(
                Mixed,
                *tuple(
                    LazyLoader(ImageSet,
                               f'path/{dset}/train.txt',
                               domain=dset,
                               T1=T1,
                               whitestripe=whitestripe)
                    for dset in DATASETS[name] if dset != heldout)),
            'val':
            LazyLoader(
                Mixed,
                *tuple(
                    LazyLoader(ImageSet,
                               f'path/{dset}/val.txt',
                               domain=dset,
                               T1=T1,
                               train=False,
                               whitestripe=whitestripe)
                    for dset in DATASETS[name] if dset != heldout)),
            'test':
            LazyLoader(ImageSet,
                       f'path/{heldout}/test.txt',
                       domain=heldout,
                       T1=T1,
                       train=False,
                       whitestripe=whitestripe)
        }
        for heldout in DATASETS[name]
    }, DOMAINS[name]


def postprocessing(pred, domain):  #TODO:remove postprocessing
    if domain == 0 or domain == 1:
        start_slice = 6
        num_selected_slice = pred.shape[0]
        pred[0:start_slice, :, :, :] = 0
        pred[(num_selected_slice - start_slice - 1):(num_selected_slice -
                                                     1), :, :, :] = 0
    else:
        start_slice = 11
        num_selected_slice = pred.shape[0]
        pred[0:start_slice, :, :, :] = 0
        pred[(num_selected_slice - start_slice - 1):(num_selected_slice -
                                                     1), :, :, :] = 0

    return pred
