import torch.nn as nn
from torch.nn import functional as F

from dassl.utils import init_network_weights

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_hidden=64, ms_class=None, ms_p=0.5, ms_a=0.1, ms_layers=[]):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = False, False, False
        self.style_layer_num = len(ms_layers)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x, style_step=False):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if "layer2" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        if "layer3" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)


class UConvNet(Backbone):

    def __init__(self, c_hidden=64, perturbation=None, uncertainty=0.5, dsu_layers=[]):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self.pertubration = perturbation(p=uncertainty) if perturbation else nn.Identity()
        self.dsu_layers = dsu_layers
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = False, False, False
        self.style_layer_num = len(dsu_layers)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x, style_step=False):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        if "layer1" in self.dsu_layers:
            x = self.pertubration(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if "layer2" in self.dsu_layers:
            x = self.pertubration(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        if "layer3" in self.dsu_layers:
            x = self.pertubration(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        if "layer4" in self.dsu_layers:
            x = self.pertubration(x)
        return x.view(x.size(0), -1)


class ConvNet_Sigma_dfed(Backbone):

    def __init__(self, c_hidden=64, ms_class=None, ms_p=0.5, ms_q=1, ms_a=0.1, ms_layers=[]):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self.mixstyle_Sigma_dfed = None
        if ms_layers:
            self.mixstyle_Sigma_dfed = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle_Sigma_dfed.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = True, True, False
        self.mu_dict = {layer : None for layer in ms_layers}
        self.sig_dict = {layer : None for layer in ms_layers}
        self.Sigma_mu_dict = {layer : None for layer in ms_layers}
        self.Sigma_sig_dict = {layer : None for layer in ms_layers}
        self.style_layer_num = len(ms_layers)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x, mu_avg=None, sig_avg=None, Sigma_mu_avg=None, Sigma_sig_avg=None, style_step=False):
        mus, sigs, Sigma_mus, Sigma_sigs = {}, {}, {}, {}
        for layer in self.ms_layers:
            mus[layer], sigs[layer], Sigma_mus[layer], Sigma_sigs[layer] = None, None, None, None
            if mu_avg != None:
                mus[layer], sigs[layer], Sigma_mus[layer], Sigma_sigs[layer] = {}, {}, {}, {}
                for neighbor_name in mu_avg:
                    mus[layer][neighbor_name] = mu_avg[neighbor_name][layer]
                    sigs[layer][neighbor_name] = sig_avg[neighbor_name][layer]
                    Sigma_mus[layer][neighbor_name] = Sigma_mu_avg[neighbor_name][layer]
                    Sigma_sigs[layer][neighbor_name] = Sigma_sig_avg[neighbor_name][layer]

        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        if "layer1" in self.ms_layers:
            if style_step:
                self.mu_dict["layer1"], self.sig_dict["layer1"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            # x = self.mixstyle_Sigma_dfed(x, mu_avg["layer1"], sig_avg["layer1"], Sigma_mu_avg["layer1"], Sigma_sig_avg["layer1"])
            x = self.mixstyle_Sigma_dfed(x, mus["layer1"], sigs["layer1"], Sigma_mus["layer1"], Sigma_sigs["layer1"])
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if "layer2" in self.ms_layers:
            if style_step:
                self.mu_dict["layer2"], self.sig_dict["layer2"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            # x = self.mixstyle_Sigma_dfed(x, mu_avg["layer1"], sig_avg["layer1"], Sigma_mu_avg["layer1"], Sigma_sig_avg["layer1"])
            x = self.mixstyle_Sigma_dfed(x, mus["layer2"], sigs["layer2"], Sigma_mus["layer2"], Sigma_sigs["layer2"])
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        if "layer3" in self.ms_layers:
            if style_step:
                self.mu_dict["layer3"], self.sig_dict["layer3"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            # x = self.mixstyle_Sigma_dfed(x, mu_avg["layer1"], sig_avg["layer1"], Sigma_mu_avg["layer1"], Sigma_sig_avg["layer1"])
            x = self.mixstyle_Sigma_dfed(x, mus["layer3"], sigs["layer3"], Sigma_mus["layer3"], Sigma_sigs["layer3"])
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

    def extract_average_statistics(self):
        mu_dict_avg, sig_dict_avg = {}, {}
        Sigma_mu_dict_avg, Sigma_sig_dict_avg = {}, {}

        for layer in self.ms_layers:
            mu_dict_avg[layer] = self.mu_dict[layer].mean(dim=0, keepdim=True)
            sig_dict_avg[layer] = self.sig_dict[layer].mean(dim=0, keepdim=True)
            Sigma_mu_dict_avg[layer] = (self.mu_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
            Sigma_sig_dict_avg[layer] = (self.sig_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()

        return mu_dict_avg, sig_dict_avg, Sigma_mu_dict_avg, Sigma_sig_dict_avg


@BACKBONE_REGISTRY.register()
def cnn_digitsdg(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = ConvNet(c_hidden=64)
    init_network_weights(model, init_type="kaiming")
    return model


@BACKBONE_REGISTRY.register()
def cnn_digitsdg_ms_l123(**kwargs):
    from dassl.modeling.ops import MixStyle

    model = ConvNet(c_hidden=64, ms_class=MixStyle, ms_layers=["layer1", "layer2", "layer3"])
    init_network_weights(model, init_type="kaiming")
    return model


@BACKBONE_REGISTRY.register()
def ucnn_digitsdg(**kwargs):
    from dassl.modeling.ops import DistributionUncertainty

    model = UConvNet(c_hidden=64, perturbation=DistributionUncertainty,
        dsu_layers=["layer1", "layer2", "layer3", "layer4"])
    init_network_weights(model, init_type="kaiming")
    return model


@BACKBONE_REGISTRY.register()
def cnn_digitsdg_Stable_l123_Sigma_dfed(**kwargs):
    from dassl.modeling.ops import StyleExplore_Sigma_dfed

    model = ConvNet_Sigma_dfed(c_hidden=64, ms_class=StyleExplore_Sigma_dfed,
        ms_layers=["layer1", "layer2", "layer3"])
    init_network_weights(model, init_type="kaiming")
    return model
