from pathlib import Path
import argparse
import sys
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random

# remove dot if debug
from .models.adv_exp_intensity import train_adversarial_examples_intensity
from .datasets.utils import get_splits, postprocessing
from .models.models import UNet, LightWeight
from .models.heads import UNetDiscriminator
from .models.discriminator import Discriminator
from .models.losses import HLoss, DSCLoss, DSCLoss_2
from .datasets.dataset import Augmentation
from .models.metrics import Critic
from .models.utils import parser_add_argument, validation_eval, test_eval, set_random_seed, check_cosine_sim
from .models.adv_exp import train_adversarial_examples
from .models.mixup import mixup_data, mixup_criterion
from .evalPerSubject import evaluatePerSubject

if __name__ == '__main__':
    parser = parser_add_argument()
    args = parser.parse_args()
    set_random_seed(42)

    splits, num_domains = get_splits('WMH_SEG',
                                     T1=args.T1,
                                     whitestripe=args.whitestripe,
                                     test_on_local=args.test_on_local)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        gpus = [int(x.strip()) for x in args.gpu.split(',')]
    for heldout in splits.keys():
        if args.single_target is not None:
            if heldout != args.single_target:
                continue
        if args.test_on_local:
            if heldout != 'Local':
                continue
        _save_dir = 'results/' + args.save_dir + '/' + heldout
        Path(_save_dir).mkdir(parents=True, exist_ok=True)
        if args.model == 'unet':
            print('Using UNet')
            model = UNet(T1=args.T1)
        dsc_loss_fn = DSCLoss(1)
        if use_cuda:
            gpu_ids = []
            for i in range(len(gpus)):
                gpu_ids.append(i)
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
        model.load_state_dict(
            torch.load('tsne_results/dannMixup_local_model.pt'))
        model.cuda()
        trainset = splits[heldout]['train']()
        if args.normal_aug:
            print('Using random augmentation')
            trainset = Augmentation(trainset,
                                    base_and_aug=args.base_and_aug,
                                    intensity_rescale=args.is_aug)
        else:
            print('Using no augmentation')
            trainset = Augmentation(trainset, do_aug=False)

        train = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            drop_last=True,
                                            num_workers=4,
                                            shuffle=True)
        out_domain = []
        out_data = []
        out_features = []
        j = 0
        for epoch in range(0, 1):
            for i, batch in enumerate(train):
                # if j == 20:
                #     break
                x = batch['img'].cuda()
                y = batch['label'].cuda()
                brain_mask = batch['mask'].bool().cuda()
                d = batch['domain'].cuda()
                model.eval()

                yhat, z, _ = model(x)
                features_np = z.data.cpu().numpy()
                domain_np = d.data.cpu().numpy()
                data_np = x.data.cpu().numpy()

                out_features.append(features_np)
                out_domain.append(domain_np[:, np.newaxis])
                out_data.append(data_np)
                j += 1

        featuers_array = np.concatenate(out_features, axis=0)
        domain_array = np.concatenate(out_domain, axis=0)
        data_array = np.concatenate(out_data, axis=0)

        np.save('tsne_results/features.npy',
                featuers_array,
                allow_pickle=False)
        np.save('tsne_results/domain.npy', domain_array, allow_pickle=False)
        np.save('tsne_results/data.npy', data_array, allow_pickle=False)
