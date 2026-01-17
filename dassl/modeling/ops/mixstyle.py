import random
from contextlib import contextmanager
import torch
import torch.nn as nn
# DDG-added
from copy import deepcopy
from random import choice


# DDG-edited
def deactivate_mixstyle(m):
    if type(m) in [MixStyle, MixStyle_dfed, MixStyle_Sigma_dfed, MixStyle_ViT]:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) in [MixStyle, MixStyle_dfed, MixStyle_Sigma_dfed, MixStyle_ViT]:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) in [MixStyle, MixStyle_dfed, MixStyle_Sigma_dfed, MixStyle_ViT]:
        m.update_mix_method("random")


def crossdomain_mixstyle(m):
    if type(m) in [MixStyle, MixStyle_dfed, MixStyle_Sigma_dfed, MixStyle_ViT]:
        m.update_mix_method("crossdomain")


# SOTS-added
def sots_mixstyle(m):
    if type(m) in [MixStyle, MixStyle_dfed, MixStyle_Sigma_dfed, MixStyle_ViT]:
        m.update_mix_method("sots")


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == "random":
        model.apply(random_mixstyle)

    elif mix == "crossdomain":
        model.apply(crossdomain_mixstyle)

    # SOTS-added
    elif mix == "sots":
        model.apply(sots_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)


class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
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
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
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

        x_normed = (x-mu) / sig

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

        # SOTS-added
        elif self.mix == "sots":
            if not shift:
                 # AdaIn the styles to the mean style
                 mu_mix, sig_mix = mu.mean(dim=0), sig.mean(dim=0)
            else:
                 # AdaIn the styles to the given style
                 mu_mix, sig_mix = mu_shift, sig_shift
            return x_normed * sig_mix + mu_mix

        else:
            raise NotImplementedError

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
class MixStyle_dfed(nn.Module):
    """MixStyle_dfed.

    Reference:
      .
    """

    def __init__(self, p=0.5, q=0.5, alpha=0.1, eps=1e-6, mix="random"):
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
            f"MixStyle_dfed(p={self.p}, q={self.q}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x, mu_avg, sig_avg):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        # DDG-edited
        mu, sig = self.extract_statistics(x)

        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if random.random() > self.q:
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

            mu2, sig2 = mu[perm], sig[perm]

        else:
            mu2, sig2 = deepcopy(mu), deepcopy(sig)
            # DDG-added
            if mu_avg != None and sig_avg != None:
                # for i in range(B):
                #     mu2[i], sig2[i] = mu_avg, sig_avg
                mu2, sig2 = mu_avg, sig_avg

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
class MixStyle_Sigma_dfed(nn.Module):
    """MixStyle_Sigma_dfed.

    Reference:
      .
    """

    def __init__(self, p=0.5, q=2/3, alpha=0.1, eps=1e-6, mix="random"):
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
            f"MixStyle_Sigma_dfed(p={self.p}, q={self.q}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

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

        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if random.random() > self.q:
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

            mu2, sig2 = mu[perm], sig[perm]

        else:
            # DDG-added
            mu2, sig2 = deepcopy(mu), deepcopy(sig)
            if mu_avg != None and sig_avg != None and Sigma_mu_avg != None and Sigma_sig_avg != None:
                for i in range(B):
                    # Inspired by DSU
                    # mu2[i] = self._reparameterize(mu_avg, Sigma_mu_avg)
                    # sig2[i] = self._reparameterize(sig_avg, Sigma_sig_avg)

                    # Style bank version
                    neighbor_choice = choice(list(mu_avg.keys()))
                    mu2[i] = self._reparameterize(mu_avg[neighbor_choice], Sigma_mu_avg[neighbor_choice])
                    sig2[i] = self._reparameterize(sig_avg[neighbor_choice], Sigma_sig_avg[neighbor_choice])

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


class MixStyle_ViT(nn.Module):
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
            f"MixStyle_ViT(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
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

        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
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

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

    # Author-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mu = x.mean(dim=[2], keepdim=True)
            var = x.var(dim=[2], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
        return mu, sig
