import torch
import argparse
import numpy as np
import random
import builtins


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=distributed.reduce_op.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_cosine_sim(task_loss, aux_loss, shared_model):

    task_grad = torch.autograd.grad(task_loss,
                                    shared_model.parameters(),
                                    only_inputs=True,
                                    retain_graph=True)
    task_grad = torch.cat([g.view(-1) for g in task_grad], dim=0)
    aux_grad = torch.autograd.grad(aux_loss,
                                   shared_model.parameters(),
                                   only_inputs=True,
                                   retain_graph=True)
    aux_grad = torch.cat([g.view(-1) for g in aux_grad], dim=0)
    sim = (task_grad * aux_grad).sum()
    print(f'sim: {sim}')
    return sim > 0


def parser_add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        help='Print log file',
                        action='store_true')
    parser.add_argument('--gpu', type=str, help='GPU idx to run', default='0')
    parser.add_argument('--save_dir',
                        type=str,
                        help='Write directory',
                        default='output')
    parser.add_argument('--model',
                        type=str,
                        choices=['unet', 'light_weight'],
                        help='model',
                        default='unet')
    parser.add_argument('--features_lr',
                        type=float,
                        help='Feature extractor learning rate',
                        default=2e-4)
    parser.add_argument('--classifier_lr',
                        type=float,
                        help='Classifier learning rate',
                        default=2e-4)
    parser.add_argument('--max_num_epochs',
                        type=int,
                        help='Max Number of epochs',
                        default=150)
    parser.add_argument('--patience',
                        type=int,
                        help='Patience of optimizer',
                        default=30)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch Size',
                        default=30)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='Weight Decay',
                        default=5e-4)
    parser.add_argument('--domain_adversary',
                        help='Discriminate domains adversarially',
                        action='store_true')
    parser.add_argument('--domain_adversary_weight',
                        type=float,
                        help='Weight for domain adversarial loss',
                        default=1.0)
    parser.add_argument('--domain_adversary_lr',
                        type=float,
                        help='Learning rate for domain adversary',
                        default=2e-4)
    parser.add_argument('--early_adversary_suppression',
                        help='Suppress adversaries 2/{1+exp{-k*p}}-1',
                        action='store_true')
    parser.add_argument('--suppression_decay',
                        type=float,
                        help='Weight to use in suppression',
                        default=3.0)
    parser.add_argument('--suppression_max',
                        type=int,
                        help='Maximum number of epoch to use suppression',
                        default=30)
    parser.add_argument('--normal_aug',
                        help='Apply random augmentation',
                        action='store_true')
    parser.add_argument('--clip_gradients',
                        help='Clip the gradient values',
                        action='store_true')
    parser.add_argument('--wandb',
                        type=str,
                        help='plot on wandb ',
                        default=None)
    parser.add_argument('--base_and_aug',
                        help='base images add aug images; default is 100% aug',
                        action='store_true')
    parser.add_argument('--entropy',
                        help='entropy loss; default is False',
                        action='store_true')
    parser.add_argument('--entropy_weight',
                        type=float,
                        help='entropy loss weight; default is 1',
                        default=0.01)
    parser.add_argument('--entropy_mask',
                        help='entropy mask default is False',
                        action='store_true')
    parser.add_argument('--save_batch', help='save batch', action='store_true')
    parser.add_argument('--use_mask', help='use mask', action='store_true')
    parser.add_argument('--lr_step',
                        type=int,
                        help='Steps between LR decrease, 80% of 80 epochs',
                        default=64)
    parser.add_argument('--momentum',
                        type=int,
                        help='Momentum for SGD',
                        default=0.9)
    parser.add_argument('--postprocess',
                        help='postprocess the predition for metric',
                        action='store_true')
    parser.add_argument('--T1',
                        help='Add T1 as second channel',
                        action='store_true')
    parser.add_argument('--is_aug',
                        help='intensity rescaling augmentation',
                        type=float,
                        default=None)
    parser.add_argument('--factor',
                        type=float,
                        help='factor of optimizer',
                        default=0.5),
    parser.add_argument('--random_seed',
                        type=int,
                        help='random seed',
                        default=42)
    parser.add_argument('--second_stage',
                        help='train the model on second stage',
                        action='store_true')
    parser.add_argument('--second_stage_epochs',
                        type=int,
                        help='epoch of second stage',
                        default=100)
    parser.add_argument(
        '--adversarial_examples',
        help='Train with examples adversarial to Feature Extractor',
        action='store_true')
    parser.add_argument('--adversarial_examples_lr',
                        type=float,
                        help='Learning rate for adversarial examples',
                        default=1e-1)
    parser.add_argument('--adversarial_examples_wd',
                        type=float,
                        help='Weight Decay for adversarial examples',
                        default=1e-4)
    parser.add_argument('--adversarial_train_steps',
                        type=int,
                        help='Steps to train adversarial examples',
                        default=50)
    parser.add_argument('--adversarial_examples_ratio',
                        type=float,
                        help='Ratio of adversarial examples',
                        default=0.5)
    parser.add_argument('--adv_blur_step',
                        type=int,
                        help='How many steps between blurring',
                        default=4)
    parser.add_argument('--adv_blur_sigma',
                        type=float,
                        help='Size of sigma in blurring',
                        default=1)
    parser.add_argument('--adv_kl_weight',
                        type=float,
                        help='Use kl loss on classes for adv examples',
                        default=1.0)
    parser.add_argument('--save_adversarial_examples',
                        help='Save examples adversarial to DANN',
                        action='store_true')
    parser.add_argument('--no_adversary_on_original',
                        help='adversary on color jitter only',
                        action='store_true')
    parser.add_argument('--classify_adv_exp',
                        help='classify adversarial examples',
                        action='store_true')
    parser.add_argument('--whitestripe',
                        help='use whitestripe normalized dataset',
                        action='store_true')
    parser.add_argument('--cosine_aux_switch',
                        help='use cosine aux switch',
                        action='store_true'),
    parser.add_argument('--test_on_local',
                        help='test on local dataset',
                        action='store_true')
    parser.add_argument('--adv_exp_is',
                        help='train adv exp based intensity',
                        action='store_true')
    parser.add_argument('--multi_input_dann',
                        help='pass multi input features to dann',
                        action='store_true')
    parser.add_argument('--single_target',
                        help='test on single target',
                        type=str,
                        default=None)
    parser.add_argument('--mixup_rate',
                        help='mixup rate training data',
                        type=float,
                        default=None)
    parser.add_argument('--mixup_example_ratio',
                        help='mixup ratio of training data',
                        type=float,
                        default=0.5)
    parser.add_argument('--classify_mixup_domain',
                        help='classify_mixup_domain',
                        action='store_true')
    parser.add_argument('--mixup_dann_features',
                        help='mixup_dann_features',
                        action='store_true')
    parser.add_argument('--docker_setups',
                        help='docker_setups',
                        action='store_true')
    return parser


def validation_eval(model,
                    dset,
                    critic,
                    dsc_loss_fn,
                    postprocessing,
                    epoch,
                    args,
                    heldout,
                    wandb=None):
    val_dsc_loss = 0
    val_acc = 0
    val_ppv = 0
    val_tpr = 0
    val_fpr = 0
    val_dsc = 0
    val_bs = 0
    val_len = 0
    val_dsc_li = 0
    val_avd_li = 0
    val_h95_li = 0
    val_recall_li = 0
    val_f1_li = 0

    with torch.no_grad():
        model.eval()
        val_metrics = critic.list_template()

        # Getting images from each subject -----------------------------------------------------------
        if args.postprocess:
            x_all_val_0 = None
            y_all_val_0 = None
            b_all_val_0 = None
            domain_val_0 = None
            x_all_val_1 = None
            y_all_val_1 = None
            b_all_val_1 = None
            domain_val_1 = None

            if heldout == 'GE3T':
                domain_check_0 = 0
                domain_check_1 = 1
            elif heldout == 'Singapore':
                domain_check_0 = 1
                domain_check_1 = 2
            elif heldout == 'Utrecht':
                domain_check_0 = 0
                domain_check_1 = 2

            for i, batch in enumerate(dset):
                if batch['domain_s'] == domain_check_0:
                    domain_val_0 = batch['domain_s']
                    if x_all_val_0 is None:
                        x_all_val_0 = batch['img']
                        y_all_val_0 = batch['label']
                        b_all_val_0 = batch['mask'].bool()
                        continue
                    x_all_val_0 = torch.cat((x_all_val_0, batch['img']), dim=0)
                    y_all_val_0 = torch.cat((y_all_val_0, batch['label']),
                                            dim=0)
                    b_all_val_0 = torch.cat(
                        (b_all_val_0, batch['mask'].bool()), dim=0)
                elif batch['domain_s'] == domain_check_1:
                    domain_val_1 = batch['domain_s']
                    if x_all_val_1 is None:
                        x_all_val_1 = batch['img']
                        y_all_val_1 = batch['label']
                        b_all_val_1 = batch['mask'].bool()
                        continue
                    x_all_val_1 = torch.cat((x_all_val_1, batch['img']), dim=0)
                    y_all_val_1 = torch.cat((y_all_val_1, batch['label']),
                                            dim=0)
                    b_all_val_1 = torch.cat(
                        (b_all_val_1, batch['mask'].bool()), dim=0)
            # ------------------------------------------------------------------------------------------------
            # for i in range(0, len(x_all_val_1)-1, 83):
            #     print(i, i+83)
            #     x = x_all_val_1[i:i+83, :, :, :]
            #     x = x.cpu().numpy()
            #     print(x.shape)
            #     for j in range(x.shape[0]):
            #         x_x = x[j, 0, :, :]
            #         plt.imsave(
            #             f'../img_1/{i}___{j}.png', x_x, origin='lower', cmap='gray')
            # ------------------------------------------------------------------------------------------------

            # Validate each subject --------------------------------------------------------------------------
            step = 0
            if domain_val_0 == 0 or domain_val_0 == 1:
                step = 48
            else:
                step = 83
            for i in range(0, len(x_all_val_0) - 1, step):
                x = x_all_val_0[i:i + step, :, :, :].cuda()
                y = y_all_val_0[i:i + step, :, :, :].cuda()
                brain_mask = b_all_val_0[i:i + step, :, :, :].cuda()
                yhat, _, _ = model(x)
                if args.use_mask:
                    yhat = yhat[brain_mask]
                    y = y[brain_mask]
                # if args.postprocess:
                #     yhat = postprocessing(yhat, domain_val_0)
                val_dsc_loss += dsc_loss_fn(yhat, y).item()

                val_metrics = critic(yhat, y, train=True)
                val_acc += val_metrics['ACC']
                val_ppv += val_metrics['PPV']
                val_tpr += val_metrics['TPR']
                val_fpr += val_metrics['FPR']
                val_dsc += val_metrics['DSC']
                val_bs += val_metrics['BS']
                if epoch % 5 == 0:
                    val_dsc_li += val_metrics['DSC_li']
                    val_avd_li += val_metrics['AVD_li']
                    val_h95_li += val_metrics['H95_li']
                    val_recall_li += val_metrics['Lesion_Recall_li']
                    val_f1_li += val_metrics['F1_li']
                val_len += 1

            if domain_val_1 == 0 or domain_val_1 == 1:
                step = 48
            else:
                step = 83
            for i in range(0, len(x_all_val_1) - 1, step):
                x = x_all_val_1[i:i + step, :, :, :].cuda()
                y = y_all_val_1[i:i + step, :, :, :].cuda()
                brain_mask = b_all_val_1[i:i + step, :, :, :].cuda()
                yhat, _, _ = model(x)
                if args.use_mask:
                    yhat = yhat[brain_mask]
                    y = y[brain_mask]
                # if args.postprocess:
                #     yhat = postprocessing(yhat, domain_val_1)
                val_dsc_loss += dsc_loss_fn(yhat, y).item()
                val_metrics = critic(yhat, y, train=True)
                val_acc += val_metrics['ACC']
                val_ppv += val_metrics['PPV']
                val_tpr += val_metrics['TPR']
                val_fpr += val_metrics['FPR']
                val_dsc += val_metrics['DSC']
                val_bs += val_metrics['BS']
                val_len += 1
            print(val_len)
        # ------------------------------------------------------------------------------------------------
        else:
            for i, batch in enumerate(dset):
                x = batch['img'].cuda()
                y = batch['label'].cuda()
                d = batch['domain']
                brain_mask = batch['mask'].bool().cuda()
                yhat, _, _ = model(x)
                if args.use_mask:
                    yhat = yhat[brain_mask]
                    y = y[brain_mask]
                val_dsc_loss += dsc_loss_fn(yhat, y).item()
                val_metrics = critic(yhat, y, train=True)
                val_acc += val_metrics['ACC']
                val_ppv += val_metrics['PPV']
                val_tpr += val_metrics['TPR']
                val_fpr += val_metrics['FPR']
                val_dsc += val_metrics['DSC']
                val_bs += val_metrics['BS']

                val_len += 1
                if wandb is not None:
                    if i % 5 == 0:
                        x_flair = x.cpu().numpy()[:, 0, :, :]
                        if x.shape[1] == 2:
                            x_t1 = x.cpu().numpy()[:, 1, :, :]
                        else:
                            x_t1 = None
                        y_groundTruth = y.cpu().numpy()[:, 0, :, :]
                        y_pred = (yhat.sigmoid() >
                                  .5).long().cpu().numpy()[:, 0, :, :]
                        d_ = d.numpy()

                        wandb.log({
                            f'{heldout}_x_flair_val': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} domain: {d_i}")
                                for d_i, img in zip(d_, x_flair)
                            ],
                            f'{heldout}_x_t1_val': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} domain: {d_i}")
                                for d_i, img in zip(d_, x_t1)
                            ] if x_t1 is not None else None,
                            f'{heldout}_y_groundTruth_val': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} domain: {d_i}")
                                for d_i, img in zip(d_, y_groundTruth)
                            ],
                            f'{heldout}_y_pred_val': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} domain: domain: {d_i}"
                                ) for d_i, img in zip(d_, y_pred)
                            ]
                        })

        val_dsc_loss = val_dsc_loss / val_len
        val_acc = val_acc / val_len
        val_ppv = val_ppv / val_len
        val_tpr = val_tpr / val_len
        val_fpr = val_fpr / val_len
        val_dsc = val_dsc / val_len
        val_bs = val_bs / val_len
        val_dict = {
            'val_dsc_loss': val_dsc_loss,
            'val_acc': val_acc,
            'val_ppv': val_ppv,
            'val_tpr': val_tpr,
            'val_fpr': val_fpr,
            'val_dsc': val_dsc,
            'val_bs': val_bs,
            'val_dsc_li': val_dsc_li,
            'val_avd_li': val_avd_li,
            'val_h95_li': val_h95_li,
            'val_recall_li': val_recall_li,
            'val_f1_li': val_f1_li
        }
        return val_dict


def test_eval(model,
              dset,
              critic,
              dsc_loss_fn,
              postprocessing,
              epoch,
              args,
              no_li_critic=True,
              wandb=None,
              heldout=None,
              best_val=False):
    test_dsc_loss = 0
    test_acc = 0
    test_ppv = 0
    test_tpr = 0
    test_fpr = 0
    test_dsc = 0
    test_bs = 0
    test_dsc_li = 0
    test_avd_li = 0
    test_h95_li = 0
    test_recall_li = 0
    test_f1_li = 0
    test_len = 0
    if not no_li_critic:
        wandb_saveImg_step = 1
    else:
        wandb_saveImg_step = 5
    with torch.no_grad():
        model.eval()
        test_metrics = critic.list_template()
        # Getting images from each subject -------------------------------------------
        if args.postprocess:
            x_all_test = None
            y_all_test = None
            b_all_test = None
            domain_test = None
            for i, batch in enumerate(dset):
                if domain_test is None:
                    domain_test = batch['domain_s'][0]
                if x_all_test is None:
                    x_all_test = batch['img']
                    y_all_test = batch['label']
                    b_all_test = batch['mask'].bool()
                    continue
                x_all_test = torch.cat((x_all_test, batch['img']), dim=0)
                y_all_test = torch.cat((y_all_test, batch['label']), dim=0)
                b_all_test = torch.cat((b_all_test, batch['mask'].bool()),
                                       dim=0)
            # --------------------------------------------------------------------------

            # Test each subject --------------------------------------------------------
            if domain_test == 0 or domain_test == 1:
                step = 48
            else:
                step = 83

            for i in range(0, len(x_all_test) - 1, step):
                x = x_all_test[i:i + step, :, :, :].cuda()
                y = y_all_test[i:i + step, :, :, :].cuda()
                brain_mask = b_all_test[i:i + step, :, :, :].cuda()
                yhat, _, _ = model(x)
                if args.use_mask:
                    yhat = yhat[brain_mask]
                    y = y[brain_mask]
                # if args.postprocess:
                #     yhat = postprocessing(yhat, domain_test)
                test_dsc_loss += dsc_loss_fn(yhat, y).item()
                test_metrics = critic(yhat, y, train=no_li_critic)
                test_acc += test_metrics['ACC']
                test_ppv += test_metrics['PPV']
                test_tpr += test_metrics['TPR']
                test_fpr += test_metrics['FPR']
                test_dsc += test_metrics['DSC']
                test_bs += test_metrics['BS']
                test_dsc_li += test_metrics['DSC_li']
                test_avd_li += test_metrics['AVD_li']
                test_h95_li += test_metrics['H95_li']
                test_recall_li += test_metrics['Lesion_Recall_li']
                test_f1_li += test_metrics['F1_li']
                test_len += 1
        # --------------------------------------------------------------------------
        else:
            for i, batch in enumerate(dset):
                x = batch['img'].cuda()
                y = batch['label'].cuda()
                brain_mask = batch['mask'].bool().cuda()
                yhat, _, _ = model(x)
                if args.use_mask:
                    yhat = yhat[brain_mask]
                    y = y[brain_mask]
                test_dsc_loss += dsc_loss_fn(yhat, y).item()
                test_metrics = critic(yhat, y, train=no_li_critic)
                test_acc += test_metrics['ACC']
                test_ppv += test_metrics['PPV']
                test_tpr += test_metrics['TPR']
                test_fpr += test_metrics['FPR']
                test_dsc += test_metrics['DSC']
                test_bs += test_metrics['BS']
                test_dsc_li += test_metrics['DSC_li']
                test_avd_li += test_metrics['AVD_li']
                test_h95_li += test_metrics['H95_li']
                test_recall_li += test_metrics['Lesion_Recall_li']
                test_f1_li += test_metrics['F1_li']
                test_len += 1

                if wandb is not None:
                    if i % wandb_saveImg_step == 0:
                        x_flair = x.cpu().numpy()[:, 0, :, :]
                        if x.shape[1] == 2:
                            x_t1 = x.cpu().numpy()[:, 1, :, :]
                        else:
                            x_t1 = None
                        y_groundTruth = y.cpu().numpy()[:, 0, :, :]
                        y_pred = (yhat.sigmoid() >
                                  .5).long().cpu().numpy()[:, 0, :, :]

                        wandb.log({
                            f'{heldout}_x_flair_test': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} note:{None if not best_val else 'best_val'}"
                                ) for img in x_flair
                            ],
                            f'{heldout}_x_t1_test': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} note:{None if not best_val else 'best_val'}"
                                ) for img in x_t1
                            ] if x_t1 is not None else None,
                            f'{heldout}_y_groundTruth_test': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} note:{None if not best_val else 'best_val'}"
                                ) for img in y_groundTruth
                            ],
                            f'{heldout}_y_pred_test': [
                                wandb.Image(
                                    img,
                                    caption=
                                    f"epoch:{epoch} iter: {i} note:{None if not best_val else 'best_val'}"
                                ) for img in y_pred
                            ]
                        })
        test_dsc_loss = test_dsc_loss / test_len
        test_acc = test_acc / test_len
        test_ppv = test_ppv / test_len
        test_tpr = test_tpr / test_len
        test_fpr = test_fpr / test_len
        test_dsc = test_dsc / test_len
        test_bs = test_bs / test_len
        test_dsc_li = test_dsc_li / test_len
        test_avd_li = test_avd_li / test_len
        test_h95_li = test_h95_li / test_len
        test_recall_li = test_recall_li / test_len
        test_f1_li = test_f1_li / test_len
        test_dict = {
            'test_dsc_loss': test_dsc_loss,
            'test_acc': test_acc,
            'test_ppv': test_ppv,
            'test_tpr': test_tpr,
            'test_fpr': test_fpr,
            'test_dsc': test_dsc,
            'test_bs': test_bs,
            'test_dsc_li': test_dsc_li,
            'test_avd_li': test_avd_li,
            'test_h95_li': test_h95_li,
            'test_recall_li': test_recall_li,
            'test_f1_li': test_f1_li
        }

        return test_dict
