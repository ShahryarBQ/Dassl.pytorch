import random
from contextlib import contextmanager
import torch
import torch.nn as nn
# DDG-added
from copy import deepcopy
from random import choice, sample


# DDG-edited
def deactivate_styleexplore(m):
    if type(m) in [StyleExplore, StyleExplore_Sigma_dfed]:
        m.set_activation_status(False)


def activate_styleexplore(m):
    if type(m) in [StyleExplore, StyleExplore_Sigma_dfed]:
        m.set_activation_status(True)


def random_styleexplore(m):
    if type(m) in [StyleExplore, StyleExplore_Sigma_dfed]:
        m.update_mix_method("random")


def crossdomain_styleexplore(m):
    if type(m) in [StyleExplore, StyleExplore_Sigma_dfed]:
        m.update_mix_method("crossdomain")


@contextmanager
def run_without_styleexplore(model):
    try:
        model.apply(deactivate_styleexplore)
        yield
    finally:
        model.apply(activate_styleexplore)


@contextmanager
def run_with_styleexplore(model, mix=None):
    if mix == "random":
        model.apply(random_styleexplore)

    elif mix == "crossdomain":
        model.apply(crossdomain_styleexplore)

    try:
        model.apply(activate_styleexplore)
        yield
    finally:
        model.apply(deactivate_styleexplore)


class StyleExplore(nn.Module):
    """StyleExplore.

    Reference:
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"StyleExplore(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    # SOTS-edited
    def forward(self, x, shift=False, mu_shift=None, sig_shift=None):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        # DDG-edited
        mu, sig = self.extract_statistics(x)

        with torch.no_grad():
            x_normed = (x-mu) / sig
        # x_normed = torch.cat((x_normed, x_normed), dim=0)

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        # DDG-added
        mu_mean = mu.mean(dim=[0], keepdim=True)
        sig_mean = sig.mean(dim=[0], keepdim=True)
        explore_idx = sample(range(B), k=B//2)
        for i in range(B):
            mu[i] = mu[i] + 3 * (mu[i] - mu_mean)
            sig[i] = sig[i] + 3 * (sig[i] - sig_mean)
        # mu = torch.cat((mu, mu_new), dim=0)
        # sig = torch.cat((sig, sig_new), dim=0)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

    # Author-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
        return mu, sig


# DDG-added
class StyleExplore_Sigma_dfed(nn.Module):
    """StyleExplore_Sigma_dfed.

    Reference:
      .
    """

    def __init__(self, p=0.5, q=1, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        # DDG-added
        self.q = q
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"StyleExplore_Sigma_dfed(p={self.p}, q={self.q}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def sqrtvar(self, x):
        x = x.mean(dim=[2, 3], keepdim=False)
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    # DDG-added (Inspired by DSU)
    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * 1.0
        return mu + epsilon * std

    def forward(self, x, mu_avg, sig_avg, Sigma_mu_avg, Sigma_sig_avg):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        # DDG-edited
        mu, sig = self.extract_statistics(x)

        with torch.no_grad():
            x_normed = (x-mu) / sig
        # x_normed = torch.cat((x_normed, x_normed), dim=0)

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        # DDG-edited
        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        explore_idx = sample(range(B), k=B//2)

        if random.random() > self.q:
            # DDG-added
            with torch.no_grad():
                sqrtvar_mu = self.sqrtvar(mu)
                sqrtvar_std = self.sqrtvar(sig)
                mu2 = self._reparameterize(mu, sqrtvar_mu)
                sig2 = self._reparameterize(sig, sqrtvar_std)
                # mu_mean = mu.mean(dim=[0], keepdim=True)
                # sig_mean = sig.mean(dim=[0], keepdim=True)
                # for i in explore_idx:
                #     mu[i] = mu[i] + 3 * (mu[i] - mu_mean)
                #     sig[i] = sig[i] + 3 * (sig[i] - sig_mean)
                #     # mu[i] = self._reparameterize(mu_avg[neighbor_choice], Sigma_mu_avg[neighbor_choice])
                #     # sig[i] = self._reparameterize(sig_avg[neighbor_choice], Sigma_sig_avg[neighbor_choice])
        else:
            # DDG-added
            if mu_avg != None and sig_avg != None and Sigma_mu_avg != None and Sigma_sig_avg != None:
                mu_new, sig_new = deepcopy(mu), deepcopy(sig)
                neighbor_choice = choice(list(mu_avg.keys()))
                shift_idx = sample(range(B), k=B//2)
                for i in shift_idx:
                    mu[i] = self._reparameterize(mu_avg[neighbor_choice], Sigma_mu_avg[neighbor_choice])
                    sig[i] = self._reparameterize(sig_avg[neighbor_choice], Sigma_sig_avg[neighbor_choice])
                # mu = torch.cat((mu, mu_new), dim=0)
                # sig = torch.cat((sig, sig_new), dim=0)

                mu_mean = mu.mean(dim=[0], keepdim=True)
                sig_mean = sig.mean(dim=[0], keepdim=True)
                for i in explore_idx:
                    mu[i] = mu[i] + 1 * (mu[i] - mu_mean)
                    sig[i] = sig[i] + 1 * (sig[i] - sig_mean)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

    # Author-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
        return mu, sig
