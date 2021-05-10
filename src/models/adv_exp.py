import scipy.ndimage
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
from torchvision import transforms


# adapted from https://discuss.pytorch.org/t/gaussian-kernel-layer/37619
def inv_norm():
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]),
        transforms.ToPILImage()
    ])


class GaussianLayer(torch.nn.Module):
    def __init__(self, sigma):

        super(GaussianLayer, self).__init__()

        self.sigma = sigma

        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(10),
            torch.nn.Conv2d(1, 1, 21, stride=1, padding=0, bias=None,
                            groups=3))

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        # keep it big, weights will tend to zero if sigma is small
        n = np.zeros((21, 21))
        n[10, 10] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.sigma)
        for _, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def enable_grads(model):
    for p in model.parameters():
        p.requires_grad = True
    model.train()


def kl_divergence(x, y):
    # q = F.softmax(q_logit, dim=1)
    # qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    # qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    # return qlogq - qlogp
    return F.kl_div(x.log(), y, reduction='batchmean')
    # return loss_f_mean(x.log(), y)


def train_adversarial_examples(x,
                               d,
                               num_domains,
                               args,
                               model,
                               domain_adversary,
                               d_loss_fn,
                               d_loss_weight,
                               dsc_loss_fn,
                               epoch=None,
                               i=None,
                               wandb=None,
                               heldout=None):
    disable_grads(model)
    disable_grads(domain_adversary)
    adv_mask = torch.torch.bernoulli(args.adversarial_examples_ratio *
                                     torch.ones(x.size(0))).bool().cuda()
    adv_x = x[adv_mask]
    d_orig = None
    args.adversarial_train_steps = random.randint(1, 5)
    wandb.log({f'{heldout}_adv_steps': args.adversarial_train_steps})
    if args.classify_adv_exp:
        d_orig = d.clone()
        d_orig[adv_mask] = d_orig[adv_mask] + num_domains

    if args.save_adversarial_examples and (not args.adv_img_saved_this_epoch):
        # Path(f'results/{args.save_dir}/adv_exp/{heldout}/{epoch}').mkdir(
        #     parents=True, exist_ok=True)
        # for q in range(adv_x.data.size(0)):
        #     fig, ax = plt.subplots()
        #     img = adv_x.data.cpu().numpy()
        #     img = np.squeeze(img[q], axis=0)
        #     plt.imshow(img,
        #                vmin=None,
        #                vmax=None,
        #                interpolation='none',
        #                origin='lower',
        #                cmap='gray')
        #     fig.savefig(
        #         f'results/{args.save_dir}/adv_exp/{heldout}/{epoch}/{i}-{q}-not.jpg'
        #     )
        #     plt.close(fig)
        x_img = adv_x.data.cpu().numpy()[:, 0, :, :]
        wandb.log({
            f'{heldout}_adv_steps':
            args.adversarial_train_steps,
            f'{heldout}_x_flair_orig':
            [wandb.Image(img, caption=f"epoch:{epoch} orig") for img in x_img]
        })

    if adv_x.nelement() > 0:
        old_class_distr, _, _ = model(adv_x)
        old_class_distr = old_class_distr.detach()
        adv_x = torch.nn.Parameter(adv_x)
        adv_x.requires_grad = True
        optim = torch.optim.Adam([adv_x],
                                 lr=args.adversarial_examples_lr,
                                 weight_decay=args.adversarial_examples_wd)
        for j in range(args.adversarial_train_steps):
            new_class_distr, z, z_up = model(adv_x)
            dhat = domain_adversary(z, z_up=z_up)
            optim.zero_grad()
            d_loss = d_loss_fn(dhat, d[adv_mask]) * d_loss_weight
            adv_dsc_loss = args.adv_dsc_weight * dsc_loss_fn(
                new_class_distr, old_class_distr)
            print(f'adv_dsc_loss:{adv_dsc_loss}')
            (d_loss + adv_dsc_loss).backward()
            if wandb is not None:
                assert heldout is not None
                wandb.log({
                    f"{heldout}_step": i,
                    f"{heldout}_adv_exp_d_loss": d_loss,
                    f"{heldout}_adv_dsc_loss": adv_dsc_loss.item()
                })
            optim.step()
            # max and min of normed tensors in dataset
            # adv_x.data = adv_x.data.clamp(-2.6653, 13.7833)
            # if j % args.adv_blur_step == 0:
            #     blur = GaussianLayer(args.adv_blur_sigma).cuda()
            #     disable_grads(blur)
            #     adv_x.data = blur(adv_x.data)
            #     optim.zero_grad()

    if args.save_adversarial_examples and (not args.adv_img_saved_this_epoch):
        x_img = adv_x.data.cpu().numpy()[:, 0, :, :]
        wandb.log({
            f'{heldout}_x_flair_adv':
            [wandb.Image(img, caption=f"epoch:{epoch} adv") for img in x_img]
        })
    x[adv_mask] = adv_x.data
    enable_grads(model)
    enable_grads(domain_adversary)

    return x