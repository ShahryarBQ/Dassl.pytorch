import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
# DDG-added
from random import choice

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        # DDG-edited
        mean, std, sqrtvar_mu, sqrtvar_std = self.extract_statistics(x)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    # DDG-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[2, 3], keepdim=False)
            std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

            sqrtvar_mu = self.sqrtvar(mean)
            sqrtvar_std = self.sqrtvar(std)

            # mean, std = mean.detach(), std.detach()
        return mean, std, sqrtvar_mu, sqrtvar_std


# DDG-added
class DistributionUncertainty_dfed(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, q=0.5, eps=1e-6):
        super(DistributionUncertainty_dfed, self).__init__()
        self.eps = eps
        self.p = p
        self.q = q
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x, mu_avg, std_avg):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        # DDG-added (influenced by MixStyle)
        B = x.size(0)

        # DDG-edited
        mean, std, sqrtvar_mu, sqrtvar_std = self.extract_statistics(x)

        # DDG-edited
        # if mu_avg != None and std_avg != None:
        #     for i in range(x.shape[0]):
        #         if np.random.random() < self.q:
        #             mean[i], std[i] = mu_avg, std_avg

        # mean_cat, std_cat = deepcopy(mean), deepcopy(std)
        # if mu_avg != None and std_avg != None:
        #     for i in range(B):
        #         if (np.random.random()) < self.q:
        #             mean_cat = torch.cat((mean_cat, mu_avg), dim=0)
        #             std_cat = torch.cat((std_cat, std_avg), dim=0)
        #     sqrtvar_mu = (mean_cat.var(dim=0, keepdim=True) + self.eps).sqrt()
        #     sqrtvar_mu = sqrtvar_mu.repeat(B, 1)
        #     sqrtvar_std = (std_cat.var(dim=0, keepdim=True) + self.eps).sqrt()
        #     sqrtvar_std = sqrtvar_std.repeat(B, 1)

        if (np.random.random()) < self.q:
            mean_cat, std_cat = deepcopy(mean), deepcopy(std)
            if mu_avg != None and std_avg != None:
                # for _ in range(B):
                #     mean_cat = torch.cat((mean_cat, mu_avg), dim=0)
                #     std_cat = torch.cat((std_cat, std_avg), dim=0)
                mean_cat = torch.cat((mean_cat, mu_avg), dim=0)
                std_cat = torch.cat((std_cat, std_avg), dim=0)
                sqrtvar_mu = (mean_cat.var(dim=0, keepdim=True) + self.eps).sqrt()
                sqrtvar_mu = sqrtvar_mu.repeat(mean.shape[0], 1)
                sqrtvar_std = (std_cat.var(dim=0, keepdim=True) + self.eps).sqrt()
                sqrtvar_std = sqrtvar_std.repeat(std.shape[0], 1)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        with torch.no_grad():
            x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
            x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    # DDG-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[2, 3], keepdim=False)
            std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

            sqrtvar_mu = self.sqrtvar(mean)
            sqrtvar_std = self.sqrtvar(std)

            # mean, std = mean.detach(), std.detach()
        return mean, std, sqrtvar_mu, sqrtvar_std


# DDG-added
class DistributionUncertainty_Sigma_dfed(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, q=0.5, eps=1e-6):
        super(DistributionUncertainty_Sigma_dfed, self).__init__()
        self.eps = eps
        self.p = p
        self.q = q
        self.factor = 1.0

        # DDG-added
        self.beta = torch.distributions.Beta(0.1, 0.1)

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x, mu_avg, std_avg, Sigma_mu_avg, Sigma_std_avg):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        # DDG-added (influenced by MixStyle)
        B = x.size(0)
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        # DDG-edited
        mean, std, sqrtvar_mu, sqrtvar_std = self.extract_statistics(x)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        if (np.random.random()) < self.q:
            # mean_cat, std_cat = deepcopy(mean), deepcopy(std)
            if mu_avg != None and std_avg != None and Sigma_mu_avg != None and Sigma_std_avg != None:
            #     idea 1
            #     for _ in range(B):
            #         mean_cat = torch.cat((mean_cat, self._reparameterize(mu_avg, Sigma_mu_avg)), dim=0)
            #         std_cat = torch.cat((std_cat, self._reparameterize(std_avg, Sigma_std_avg)), dim=0)
            #     sqrtvar_mu = (mean_cat.var(dim=0, keepdim=True) + self.eps).sqrt()
            #     sqrtvar_mu = sqrtvar_mu.repeat(B, 1)
            #     sqrtvar_std = (std_cat.var(dim=0, keepdim=True) + self.eps).sqrt()
            #     sqrtvar_std = sqrtvar_std.repeat(B, 1)

                # sqrtvar_mu = Sigma_mu_avg
                # sqrt_std = Sigma_std_avg

                # idea 2
                # mu_bar, std_bar, sqrtvar_mu, sqrtvar_std = 0, 0, 0, 0
                # for name in mu_avg:
                #     mu_bar += mu_avg[name] / len(mu_avg)
                #     std_bar += std_avg[name] / len(std_avg)
                # for name in Sigma_mu_avg:
                #     sqrtvar_mu += (torch.pow(Sigma_mu_avg[name], 2) + torch.pow(mu_avg[name], 2) - torch.pow(mu_bar, 2)) / len(Sigma_mu_avg)
                #     sqrtvar_std += (torch.pow(Sigma_std_avg[name], 2) + torch.pow(std_avg[name], 2) - torch.pow(std_bar, 2)) / len(Sigma_std_avg)
                # sqrtvar_mu = sqrtvar_mu.sqrt()
                # sqrtvar_std = sqrtvar_std.sqrt()

                # idea 3
                # for i in range(B):
                #     neighbor_choice = choice(list(mu_avg.keys()))
                #     beta[i] = self._reparameterize(mean[i], Sigma_mu_avg[neighbor_choice])
                #     gamma[i] = self._reparameterize(std[i], Sigma_std_avg[neighbor_choice])

                # idea 4
                # sqrtvar_mu /= (len(Sigma_mu_avg) + 1)
                # sqrtvar_std /= (len(Sigma_std_avg) + 1)
                # for name in mu_avg:
                #     sqrtvar_mu += Sigma_mu_avg[name] / (len(Sigma_mu_avg) + 1)
                #     sqrtvar_std += Sigma_std_avg[name] / (len(Sigma_std_avg) + 1)

                # idea 5
                # for name in mu_avg:
                #     sqrtvar_mu += Sigma_mu_avg[name].repeat(B, 1)
                #     sqrtvar_std += Sigma_std_avg[name].repeat(B, 1)

                # idea 6
                # for i in range(B):
                #     neighbor_choice = choice(list(mu_avg.keys()))
                #     beta[i] = self._reparameterize(mean[i], torch.linalg.vector_norm(Sigma_mu_avg[neighbor_choice]) / torch.linalg.vector_norm(sqrtvar_mu[0]) * sqrtvar_mu[0])
                #     gamma[i] = self._reparameterize(std[i], torch.linalg.vector_norm(Sigma_std_avg[neighbor_choice]) / torch.linalg.vector_norm(sqrtvar_std[0]) * sqrtvar_std[0])

                # idea 7
                # neighbor_choice = choice(list(mu_avg.keys()))
                # beta = self._reparameterize(mean, torch.linalg.vector_norm(Sigma_mu_avg[neighbor_choice]) / torch.linalg.vector_norm(sqrtvar_mu) * sqrtvar_mu)
                # gamma = self._reparameterize(std, torch.linalg.vector_norm(Sigma_std_avg[neighbor_choice]) / torch.linalg.vector_norm(sqrtvar_std) * sqrtvar_std)

                # idea 8
                # for name in mu_avg:
                #     sqrtvar_mu += torch.linalg.vector_norm(sqrtvar_mu[0]) / torch.linalg.vector_norm(Sigma_mu_avg[name]) * Sigma_mu_avg[name].repeat(B, 1)
                #     sqrtvar_std += torch.linalg.vector_norm(sqrtvar_std[0]) / torch.linalg.vector_norm(Sigma_std_avg[name]) * Sigma_std_avg[name].repeat(B, 1)

                # idea 9
                for i in range(B):
                    neighbor_choice = choice(list(mu_avg.keys()))
                    beta[i] = self._reparameterize(mu_avg[neighbor_choice], Sigma_mu_avg[neighbor_choice])
                    gamma[i] = self._reparameterize(std_avg[neighbor_choice], Sigma_std_avg[neighbor_choice])

                # idea 10
                # for i in range(B):
                #     neighbor_choice = choice(list(mu_avg.keys()))
                #     beta[i] = self._reparameterize(mu_avg[neighbor_choice], Sigma_mu_avg[neighbor_choice])
                #     gamma[i] = self._reparameterize(std_avg[neighbor_choice], Sigma_std_avg[neighbor_choice])
                # beta = mean.reshape(x.shape[0], x.shape[1], 1, 1) * lmda + beta.reshape(x.shape[0], x.shape[1], 1, 1) * (1-lmda)
                # gamma = std.reshape(x.shape[0], x.shape[1], 1, 1) * lmda + gamma.reshape(x.shape[0], x.shape[1], 1, 1) * (1-lmda)

                # idea 11
                # for i in range(B):
                #     neighbor_choice = choice(list(mu_avg.keys()))
                #     beta[i] = self._reparameterize(mu_avg[neighbor_choice], Sigma_mu_avg[neighbor_choice])
                #     gamma[i] = self._reparameterize(std_avg[neighbor_choice], Sigma_std_avg[neighbor_choice])
                #     if np.random.random() < 0.5:
                #         beta[i] = beta[i] + 3 * (beta[i] - mean[i])
                #         gamma[i] = gamma[i] + 3 * (gamma[i] - std[i])
                # beta = mean.reshape(x.shape[0], x.shape[1], 1, 1) * lmda + beta.reshape(x.shape[0], x.shape[1], 1, 1) * (1-lmda)
                # gamma = std.reshape(x.shape[0], x.shape[1], 1, 1) * lmda + gamma.reshape(x.shape[0], x.shape[1], 1, 1) * (1-lmda)

                # idea 12
                # for i in range(B):
                #     neighbor_choice = choice(list(mu_avg.keys()))
                #     beta[i] = self._reparameterize(mu_avg[neighbor_choice], sqrtvar_mu[0])
                #     gamma[i] = self._reparameterize(std_avg[neighbor_choice], sqrtvar_std[0])

        # beta = self._reparameterize(mean, sqrtvar_mu)
        # gamma = self._reparameterize(std, sqrtvar_std)

        with torch.no_grad():
            x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
            x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

    # DDG-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[2, 3], keepdim=False)
            std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

            sqrtvar_mu = self.sqrtvar(mean)
            sqrtvar_std = self.sqrtvar(std)

        return mean, std, sqrtvar_mu, sqrtvar_std


class DistributionUncertainty_ViT(nn.Module):
    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty_ViT, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        # DDG-edited
        mean, std, sqrtvar_mu, sqrtvar_std = self.extract_statistics(x)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1)) / std.reshape(x.shape[0], x.shape[1], 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1) + beta.reshape(x.shape[0], x.shape[1], 1)

        return x

    # DDG-added
    def extract_statistics(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[2], keepdim=False)
            std = (x.var(dim=[2], keepdim=False) + self.eps).sqrt()

            sqrtvar_mu = self.sqrtvar(mean)
            sqrtvar_std = self.sqrtvar(std)

            # mean, std = mean.detach(), std.detach()
        return mean, std, sqrtvar_mu, sqrtvar_std
