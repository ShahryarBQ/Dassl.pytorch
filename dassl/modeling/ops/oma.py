import random
from contextlib import contextmanager
import torch
import torch.nn as nn
# DDG-added
from copy import deepcopy


# DDG-edited
def deactivate_oma(m):
    if type(m) in [OMA, OMA_dfed, OMA_Sigma_dfed]:
        m.set_activation_status(False)


def activate_oma(m):
    if type(m) in [OMA, OMA_dfed, OMA_Sigma_dfed]:
        m.set_activation_status(True)


def random_oma(m):
    if type(m) in [OMA, OMA_dfed, OMA_Sigma_dfed]:
        m.update_mix_method("random")


def crossdomain_oma(m):
    if type(m) in [OMA, OMA_dfed, OMA_Sigma_dfed]:
        m.update_mix_method("crossdomain")


@contextmanager
def run_without_oma(model):
    # Assume OMA was initially activated
    try:
        model.apply(deactivate_oma)
        yield
    finally:
        model.apply(activate_oma)


@contextmanager
def run_with_oma(model, mix=None):
    # Assume OMA was initially deactivated
    if mix == "random":
        model.apply(random_oma)

    elif mix == "crossdomain":
        model.apply(crossdomain_oma)

    try:
        model.apply(activate_oma)
        yield
    finally:
        model.apply(deactivate_oma)


class OMA(nn.Module):
    """OMA.

    Reference:
      StableFDG
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random", style_explore=3):
        """
        Args:
          p (float): probability of using OMA.
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

        # DDG-added
        self.style_explore = style_explore

    def __repr__(self):
        return (
            f"OMA(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

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

        # DDG-edited
        mu_mean = mu.mean(dim=0, keepdim=True)
        sig_mean = sig.mean(dim=0, keepdim=True)
        mu_new = mu + self.style_explore * (mu - mu_mean)
        sig_new = sig + self.style_explore * (sig - sig_mean)

        mu_mix = mu*lmda + mu_new * (1-lmda)
        sig_mix = sig*lmda + sig_new * (1-lmda)

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
class OMA_dfed(nn.Module):
    """OMA_dfed.

    Reference:
      .
    """

    def __init__(self, p=0.5, q=0.5, alpha=0.1, eps=1e-6, mix="random", style_explore=3):
        """
        Args:
          p (float): probability of using OMA.
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

        # DDG-added
        self.style_explore = style_explore

    def __repr__(self):
        return (
            f"OMA_dfed(p={self.p}, q={self.q}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
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

        # DDG-edited
        if random.random() > self.q:
            mu_mean = mu.mean(dim=0, keepdim=True)
            sig_mean = sig.mean(dim=0, keepdim=True)
            mu_new = mu + self.style_explore * (mu - mu_mean)
            sig_new = sig + self.style_explore * (sig - sig_mean)
        else:
            mu_new, sig_new = deepcopy(mu), deepcopy(sig)
            if mu_avg != None and sig_avg != None:
                mu_new = mu + self.style_explore * (mu - mu_avg)
                sig_new = sig + self.style_explore * (sig - sig_avg)

        mu_mix = mu*lmda + mu_new * (1-lmda)
        sig_mix = sig*lmda + sig_new * (1-lmda)

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
class OMA_Sigma_dfed(nn.Module):
    """OMA_dfed.

    Reference:
      .
    """

    def __init__(self, p=0.5, q=0.5, alpha=0.1, eps=1e-6, mix="random", style_explore=3):
        """
        Args:
          p (float): probability of using OMA.
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

        # DDG-added
        self.style_explore = style_explore

    def __repr__(self):
        return (
            f"OMA_Sigma_dfed(p={self.p}, q={self.q}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
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

        # DDG-edited
        if random.random() > self.q:
            mu_mean = mu.mean(dim=0, keepdim=True)
            sig_mean = sig.mean(dim=0, keepdim=True)
            mu_new = mu + self.style_explore * (mu - mu_mean)
            sig_new = sig + self.style_explore * (sig - sig_mean)
        else:
            mu_new, sig_new = deepcopy(mu), deepcopy(sig)
            if mu_avg != None and sig_avg != None and Sigma_mu_avg != None and Sigma_sig_avg != None:
                mu_new = mu + self.style_explore * (mu - self._reparameterize(mu_avg, Sigma_mu_avg))
                sig_new = sig + self.style_explore * (sig - self._reparameterize(sig_avg, Sigma_sig_avg))

        mu_mix = mu*lmda + mu_new * (1-lmda)
        sig_mix = sig*lmda + sig_new * (1-lmda)

        return x_normed*sig_mix + mu_mix

    # Author-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
        return mu, sig
