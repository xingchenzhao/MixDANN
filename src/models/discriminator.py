import torch
"""
GRL adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf
"""


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, reverse=True):
        ctx.beta = beta
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.beta), None, None
        else:
            return (grad_output * ctx.beta), None, None


def grad_reverse(x, beta=1.0, reverse=True):
    return GradReverse.apply(x, beta, reverse)


class Discriminator(torch.nn.Module):
    def __init__(self, head, reverse=True):

        super(Discriminator, self).__init__()
        self.head = head
        self.beta = 0.0
        self.reverse = reverse

    def set_beta(self, beta):
        self.beta = beta

    def forward(self, z, z_up=None, use_grad_reverse=True):
        if use_grad_reverse:
            z = grad_reverse(z, self.beta, reverse=True)
            if z_up is not None:
                z_up = grad_reverse(z_up, self.beta, reverse=True)
        x = self.head(z, z_up=z_up)
        return x
