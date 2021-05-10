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
from .datasets.utils import get_splits, postprocessing
from .models.models import UNet, LightWeight
from .models.heads import UNetDiscriminator
from .models.discriminator import Discriminator
from .models.losses import HLoss, DSCLoss
from .datasets.dataset import Augmentation
from .models.metrics import Critic
from .models.utils import parser_add_argument, validation_eval, test_eval, set_random_seed, check_cosine_sim
from .models.mixup import mixup_data, mixup_criterion
from .evalPerSubject import evaluatePerSubject
'''
This code includes both Domain Discriminator and MixUp. You can focus on Domain Discriminator, and ignore the mixup
'''
if __name__ == '__main__':
    parser = parser_add_argument()
    args = parser.parse_args()
    set_random_seed(args.random_seed)
    if args.wandb is not None:  #wandb is a tool like tensorboard, tracking loss and accuracy. Defaults is None
        import wandb
        wandb.init(project=args.wandb, name=args.save_dir)
    else:
        wandb = None

    splits, num_domains = get_splits(
        'WMH_SEG',  #get data of different domains
        T1=args.T1,
        whitestripe=args.whitestripe,
        test_on_local=args.test_on_local)

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        gpus = [int(x.strip()) for x in args.gpu.split(',')]

    for heldout in splits.keys(
    ):  #heldout is the target domain. For each target domain, we train the model on source domains and test the model on the target domain.
        if args.single_target is not None:  #it is not important, and you can ignore this line.
            if heldout != args.single_target:
                continue
        if args.test_on_local:  #it is not important, and you can ignore this line.
            if heldout != 'Local':
                continue
        _save_dir = 'results/' + args.save_dir + '/' + heldout
        Path(_save_dir).mkdir(parents=True, exist_ok=True)
        if not args.verbose:
            sys.stdout = open(f'{_save_dir}/log.txt', 'w')
        print('args: ', args)
        ########################Initialze UNet and Domain Discriminator(DANN)#########################
        lr_groups = []
        #Because we have different learning rate for unet and domain discriminator, so we create a list of lr here.
        if args.model == 'unet':
            print('Using UNet')
            model = UNet(T1=args.T1)  #create a UNet
            DiscHead = UNetDiscriminator  #create a Domain Discriminator
        else:
            raise Exception(f'{args.model} not supported.')
        dsc_loss_fn = DSCLoss(1)

        if use_cuda:  #we make our model dataParallel, so that we can run the model on multiple gpus.
            gpu_ids = []
            for i in range(len(gpus)):
                gpu_ids.append(i)
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        lr_groups.append((model.parameters(), args.features_lr))

        if args.domain_adversary:
            print('Using domain adversary')
            '''initialize domain discriminator.
            By default, the number of domains that the DANN need to classify is 2.
            We can add a mixup_domain to classify which we will create later.'''
            domain_adversary = Discriminator(
                DiscHead(num_domains if not args.classify_mixup_domain else
                         num_domains + 1))
            if use_cuda:
                domain_adversary = torch.nn.DataParallel(domain_adversary,
                                                         device_ids=gpu_ids)
            domain_adversary = domain_adversary.cuda()
            lr_groups.append(
                (domain_adversary.parameters(), args.domain_adversary_lr))
            d_loss_fn = torch.nn.CrossEntropyLoss()

        optimizers = [
            torch.optim.Adam(params, lr, weight_decay=args.weight_decay)
            for params, lr in lr_groups
        ]

        schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                       patience=args.patience,
                                                       factor=args.factor)
            for optim in optimizers
        ]
        ###########################Creating Dataset#####################################
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

        valset = Augmentation(splits[heldout]['val'](), do_aug=False)
        val = torch.utils.data.DataLoader(
            valset,
            batch_size=1 if args.postprocess else args.batch_size,
            drop_last=False,
            num_workers=4,
            shuffle=False if args.postprocess else True)
        testset = Augmentation(splits[heldout]['test'](), do_aug=False)
        test = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=4,
            shuffle=False if args.postprocess else True)

        critic = Critic()
        ############################Train, Validate, Test #########################
        print(f'Starting {heldout}...')
        best_val_dsc_loss = None
        best_val_dsc = 0
        best_loss = None
        best_epoch = 0
        anomaly = False

        if args.wandb is not None:
            wandb.watch(model, log='all')
        for epoch in range(0, args.max_num_epochs):
            running_loss_dsc_loss = 0.0
            running_loss_model = 0.0
            running_loss_domain = 0.0
            running_loss_entropy = 0.0
            running_metric_dsc = 0.0

            p = epoch / args.max_num_epochs
            suppression = (2.0 / (1. + np.exp(-args.suppression_decay * p)) -
                           1)
            print(
                f'suppression={suppression:.4f}'
            )  #This is a suppression for the DANN loss. We want to supress the inital noise of the DANN loss.

            if args.domain_adversary:
                beta = 1 * args.domain_adversary_weight
                if args.early_adversary_suppression:
                    beta *= suppression
                print(f'beta={beta:.4f}')
                domain_adversary.module.set_beta(
                    beta)  #set the suppression for DANN.

            metrics = critic.list_template()
            dsc_counter = 0
            for i, batch in enumerate(train):
                x = batch['img'].cuda()
                y = batch['label'].cuda()
                brain_mask = batch['mask'].bool().cuda()
                d = batch['domain'].cuda()
                print(f'batch={i}')
                do_mixup = False

                for optim in optimizers:
                    optim.zero_grad()
                if args.mixup_rate is not None:  #You can ignore MixUp
                    if random.random() > (1 - args.mixup_example_ratio):
                        print('Do mixup')
                        do_mixup = True
                        x, y_a, y_b, d_a, d_b, lam = mixup_data(
                            x, y, d, alpha=args.mixup_rate)
                        if args.classify_mixup_domain:
                            d = torch.empty_like(d).fill_(num_domains).cuda()
                model.train()
                if args.domain_adversary:  #yhat is the prediction from the UNet, z and z_up is the features of early conv layers of UNet.
                    yhat, z, z_up = model(x)
                else:
                    yhat, _, _ = model(x)
                if args.use_mask:  #use_mask is false by default.
                    yhat = yhat[brain_mask]
                    y = y[brain_mask]
                if do_mixup:  #ignore mixup
                    dsc_loss = mixup_criterion(dsc_loss_fn, yhat, y_a, y_b,
                                               lam)
                else:
                    dsc_loss = dsc_loss_fn(yhat, y)
                    metrics = critic(yhat, y, train=True)
                    dsc_counter += 1

                print(f'dsc_loss={dsc_loss.item():.4f}')
                print(f'metrics= {metrics}')

                if args.entropy:  #entropy loss a potential method to increase performance, but you can ignore this now.
                    entropy_loss = HLoss()
                    ent_loss = entropy_loss(yhat)  # yhat_
                    print(f'ent_loss={ent_loss:.4f}')
                    ent_weight = args.entropy_weight
                    ent_loss_sp = ent_loss * (suppression * ent_weight)
                    print(f'ent_loss_sp={ent_loss_sp:.4f}')

                if args.domain_adversary:
                    domain_adversary.train()
                    if args.multi_input_dann:  #if we want to use both z and z_up as inputs to DANN.
                        dhat = domain_adversary(z, z_up=z_up)
                    else:
                        if not do_mixup and args.mixup_dann_features:  #ignore mixup
                            z, _, _, d_a, d_b, lam = mixup_data(
                                z, y, d, alpha=args.mixup_rate)
                        dhat = domain_adversary(
                            z)  #if we want to only use z as an input to DANN.
                    if dhat.nelement() != 0:
                        if do_mixup and not args.classify_mixup_domain:  #ignore mixup
                            d_loss = mixup_criterion(d_loss_fn, dhat, d_a, d_b,
                                                     lam)
                        elif not do_mixup and args.mixup_dann_features and not args.classify_mixup_domain:  #ignore mixup.
                            d_loss = mixup_criterion(d_loss_fn, dhat, d_a, d_b,
                                                     lam)
                        else:
                            d_loss = d_loss_fn(dhat, d)
                        print(f'domain_loss={d_loss:.4f}')
                    else:
                        d_loss = 0
                loss = 0
                loss += dsc_loss
                if args.domain_adversary:
                    loss += d_loss
                print(f'model_loss={loss.item():.4f}')

                if torch.isnan(loss):
                    anomaly = True
                    break

                running_loss_model += loss.item()
                running_loss_dsc_loss += dsc_loss.item()
                if args.domain_adversary:
                    running_loss_domain += d_loss.item()
                if args.entropy:
                    running_loss_entropy += ent_loss_sp.item()
                if not do_mixup:
                    running_metric_dsc += metrics['DSC']

                loss.backward()
                for optim in optimizers:
                    optim.step()

            if anomaly:
                print('Found anomaly. Terminating. ')
                break

            epoch_loss_model = running_loss_model / i
            print(f'epoch{epoch}_model_loss={epoch_loss_model:.4f}')
            epoch_loss_dsc = running_loss_dsc_loss / i
            print(f'epoch{epoch}_dsc_loss={epoch_loss_dsc:.4f}')
            if args.domain_adversary:
                epoch_loss_domain = running_loss_domain / i
                print(f'epoch{epoch}_domain_loss={epoch_loss_domain:.4f}')
            if args.entropy:
                epoch_loss_entropy = running_loss_entropy / i
                print(f'epoch{epoch}_entropy_loss={epoch_loss_entropy:.4f}')
            if metrics is not None:
                if dsc_counter > 0:
                    epoch_metric_dsc = running_metric_dsc / dsc_counter
                else:
                    epoch_metric_dsc = 0
                print(f'epoch{epoch}_dsc={epoch_metric_dsc:.4f}')

            print(f'Starting Validation...')  #Validation the model.
            val_metrics = validation_eval(model,
                                          val,
                                          critic,
                                          dsc_loss_fn,
                                          postprocessing,
                                          epoch,
                                          args,
                                          heldout,
                                          wandb=wandb)

            if best_val_dsc_loss is None or val_metrics[
                    'val_dsc_loss'] < best_val_dsc_loss:
                torch.save(model.state_dict(), f'{_save_dir}/model.pt')
                best_val_dsc_loss = val_metrics['val_dsc_loss']
                best_val_dsc = val_metrics['val_dsc']
                best_epoch = epoch
            for scheduler in schedulers:
                scheduler.step(val_metrics['val_dsc_loss'])

            print(f'validation_dsc_loss={val_metrics["val_dsc_loss"]:.4f}')
            print(f'validation_dsc={val_metrics["val_dsc"]:.4f}')
            print(f'Finished Validation')
            print(f'Finished epoch{epoch}')

            print(f'Starting Eval_test...')
            test_metrics = test_eval(model,
                                     test,
                                     critic,
                                     dsc_loss_fn,
                                     postprocessing,
                                     epoch,
                                     args,
                                     wandb=wandb,
                                     heldout=heldout)
            print(f'eval_test_dsc_loss={test_metrics["test_dsc_loss"]:.4f}')
            print(f'eval_test_dsc={test_metrics["test_dsc"]:.4f}')

            if args.wandb is not None:
                wandb.log({
                    f"{heldout}_epoch":
                    epoch,
                    f"{heldout}_dsc_loss":
                    epoch_loss_dsc,
                    f"{heldout}_domain_loss":
                    epoch_loss_domain if (args.domain_adversary) else None,
                    f"{heldout}_entropy_loss":
                    epoch_loss_entropy if args.entropy else None,
                    f"{heldout}_model_loss":
                    epoch_loss_model,
                    f"{heldout}_DSC":
                    epoch_metric_dsc,
                    f"{heldout}_validation_dsc_loss":
                    val_metrics['val_dsc_loss'],
                    f"{heldout}_validation_DSC":
                    val_metrics['val_dsc'],
                    f"{heldout}_eval_test_dsc_loss":
                    test_metrics['test_dsc_loss'],
                    f"{heldout}_eval_test_dsc":
                    test_metrics['test_dsc'],
                    f"{heldout}_suppression":
                    suppression if args.domain_adversary else 0,
                    f"{heldout}_current_lr":
                    optimizers[0].param_groups[0]['lr']
                })

        #----------------------------------------------------------------------------------------------#
        print('Starting testing...')
        last_test_metrics = test_eval(model,
                                      test,
                                      critic,
                                      dsc_loss_fn,
                                      postprocessing,
                                      epoch,
                                      args,
                                      no_li_critic=True,
                                      heldout=heldout,
                                      wandb=wandb)
        torch.save(model.state_dict(), f'{_save_dir}/last_epoch_model.pt')

        model.load_state_dict(torch.load(f'{_save_dir}/model.pt'))

        best_val_test_metrics = test_eval(model,
                                          test,
                                          critic,
                                          dsc_loss_fn,
                                          postprocessing,
                                          epoch,
                                          args,
                                          no_li_critic=True,
                                          heldout=heldout,
                                          wandb=wandb,
                                          best_val=True)
        #
        with open(f'results/{args.save_dir}/results.txt', 'a') as res:
            res.write(
                f'\n{heldout}:\n best_val_test_metrics: {best_val_test_metrics} \n\n last_epoch_model_test_metrics: {last_test_metrics} \n\n test_metrics:{test_metrics} \n'
            )

        print(f'best_validation_dsc={best_val_dsc:.4f}')
        print(f'best_epoch={best_epoch}')
        print('Finished testing')
        print(f'Finished {heldout}')
        if args.wandb is not None:
            wandb.log({
                f"{heldout}_best_validation_dsc":
                best_val_dsc,
                f"{heldout}_best_epoch":
                best_epoch,
                f"{heldout}_test_dsc_loss":
                test_metrics['test_dsc_loss'],
                f"{heldout}_test_DSC":
                test_metrics['test_dsc'],
                f"{heldout}_test_DSC_li":
                test_metrics['test_dsc_li'],
                f"{heldout}_test_AVD_li":
                test_metrics['test_avd_li'],
                f"{heldout}_test_H95_li":
                test_metrics['test_h95_li'],
                f"{heldout}_test_Lesion_Recall_li":
                test_metrics['test_recall_li'],
                f"{heldout}_test_F1_li":
                test_metrics['test_f1_li'],
                f"{heldout}_last_test_dsc_loss":
                last_test_metrics['test_dsc_loss'],
                f"{heldout}_last_test_dsc":
                last_test_metrics['test_dsc'],
                f"{heldout}_last_test_dsc_li":
                last_test_metrics['test_dsc_li'],
                f"{heldout}_last_test_avd_li":
                last_test_metrics['test_avd_li'],
                f"{heldout}_last_test_h95_li":
                last_test_metrics['test_h95_li'],
                f"{heldout}_last_test_lesion_recall_li":
                last_test_metrics['test_recall_li'],
                f"{heldout}_last_test_f1_li":
                last_test_metrics['test_f1_li']
            })
        #-------Evaluate final model using WMH_SEG_Challenge code ---
        # model.load_state_dict(torch.load(f'{_save_dir}/last_epoch_model.pt'))
        # evaluatePerSubject(model,
        #                    heldout,
        #                    wandb=wandb,
        #                    T1=args.T1,
        #                    save_dir=_save_dir)
