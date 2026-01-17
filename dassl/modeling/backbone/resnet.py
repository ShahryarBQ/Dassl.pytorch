import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = ms_class(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self._init_params()

        # DDG-added
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = False, False, False
        self.style_layer_num = len(ms_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # SOTS-edited
    def featuremaps(self, x, shift=False, mu_shift_dict=None, sig_shift_dict=None, style_step=False):
        # SOTS-added
        if mu_shift_dict == None and sig_shift_dict == None:
            mu_shift_dict = {"layer1": None, "layer2": None, "layer3": None}
            sig_shift_dict = {"layer1": None, "layer2": None, "layer3": None}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            # SOTS-edited
            x = self.mixstyle(x, shift, mu_shift_dict["layer1"], sig_shift_dict["layer1"])
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            # SOTS-edited
            x = self.mixstyle(x, shift, mu_shift_dict["layer2"], sig_shift_dict["layer2"])
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            # SOTS-edited
            x = self.mixstyle(x, shift, mu_shift_dict["layer3"], sig_shift_dict["layer3"])
        return self.layer4(x)

    # SOTS-edited
    def forward(self, x, shift=False, mu_shift_dict=None, sig_shift_dict=None, style_step=False):
        # SOTS-edited
        f = self.featuremaps(x, shift, mu_shift_dict, sig_shift_dict, style_step)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


class ResNet_dfed(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_q=0.5,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle_dfed = None
        if ms_layers:
            self.mixstyle_dfed = ms_class(p=ms_p, q=ms_q, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle_dfed.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self._init_params()

        # DDG-added
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = True, False, False
        self.mu_dict = {layer : None for layer in ms_layers}
        self.sig_dict = {layer : None for layer in ms_layers}
        self.style_layer_num = len(ms_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # DDG-edited
    def featuremaps(self, x, mu_avg=None, sig_avg=None, style_step=False):
        # DDG-added
        if mu_avg == None and sig_avg == None:
            mu_avg = {layer : None for layer in self.ms_layers}
            sig_avg = {layer : None for layer in self.ms_layers}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            # DDG-edited
            if style_step:
                self.mu_dict["layer1"], self.sig_dict["layer1"] = self.mixstyle_dfed.extract_statistics(x)
            x = self.mixstyle_dfed(x, mu_avg["layer1"], sig_avg["layer1"])
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            # DDG-edited
            if style_step:
                self.mu_dict["layer2"], self.sig_dict["layer2"] = self.mixstyle_dfed.extract_statistics(x)
            x = self.mixstyle_dfed(x, mu_avg["layer2"], sig_avg["layer2"])
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            # DDG-edited
            if style_step:
                self.mu_dict["layer3"], self.sig_dict["layer3"] = self.mixstyle_dfed.extract_statistics(x)
            x = self.mixstyle_dfed(x, mu_avg["layer3"], sig_avg["layer3"])
        return self.layer4(x)

    # DDG-edited
    def forward(self, x, mu_avg=None, sig_avg=None, style_step=False):
        # DDG-edited
        f = self.featuremaps(x, mu_avg, sig_avg, style_step)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)

    # Author-added
    def extract_average_statistics(self):
        mu_dict_avg, sig_dict_avg = {}, {}
        Sigma_mu_dict_avg, Sigma_sig_dict_avg = {}, {}

        for layer in self.ms_layers:
            mu_dict_avg[layer] = self.mu_dict[layer].mean(dim=0, keepdim=True)
            sig_dict_avg[layer] = self.sig_dict[layer].mean(dim=0, keepdim=True)
            Sigma_mu_dict_avg[layer] = (self.mu_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
            Sigma_sig_dict_avg[layer] = (self.sig_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()

        return mu_dict_avg, sig_dict_avg, Sigma_mu_dict_avg, Sigma_sig_dict_avg

    # DDG-added
    def extract_full_statistics(self):
        return self.mu_dict, self.sig_dict


class ResNet_Sigma_dfed(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_q=1,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle_Sigma_dfed = None
        if ms_layers:
            self.mixstyle_Sigma_dfed = ms_class(p=ms_p, q=ms_q, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle_Sigma_dfed.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self._init_params()

        # DDG-added
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = True, True, False
        self.mu_dict = {layer : None for layer in ms_layers}
        self.sig_dict = {layer : None for layer in ms_layers}
        self.Sigma_mu_dict = {layer : None for layer in ms_layers}
        self.Sigma_sig_dict = {layer : None for layer in ms_layers}
        self.style_layer_num = len(ms_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # DDG-edited
    def featuremaps(self, x, mu_avg=None, sig_avg=None, Sigma_mu_avg=None, Sigma_sig_avg=None, style_step=False):
        # DDG-added
        if mu_avg == None and sig_avg == None and Sigma_mu_avg == None and Sigma_sig_avg == None:
            # mu_avg = {layer : None for layer in self.ms_layers}
            # sig_avg = {layer : None for layer in self.ms_layers}
            # Sigma_mu_avg = {layer : None for layer in self.ms_layers}
            # Sigma_sig_avg = {layer : None for layer in self.ms_layers}
            pass
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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            # DDG-edited
            if style_step:
                self.mu_dict["layer1"], self.sig_dict["layer1"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            # x = self.mixstyle_Sigma_dfed(x, mu_avg["layer1"], sig_avg["layer1"], Sigma_mu_avg["layer1"], Sigma_sig_avg["layer1"])
            x = self.mixstyle_Sigma_dfed(x, mus["layer1"], sigs["layer1"], Sigma_mus["layer1"], Sigma_sigs["layer1"])
        x = self.layer2(x)
        if "layer2" in self.ms_layers:
            # DDG-edited
            if style_step:
                self.mu_dict["layer2"], self.sig_dict["layer2"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            # x = self.mixstyle_Sigma_dfed(x, mu_avg["layer2"], sig_avg["layer2"], Sigma_mu_avg["layer2"], Sigma_sig_avg["layer2"])
            x = self.mixstyle_Sigma_dfed(x, mus["layer2"], sigs["layer2"], Sigma_mus["layer2"], Sigma_sigs["layer2"])
        x = self.layer3(x)
        if "layer3" in self.ms_layers:
            # DDG-edited
            if style_step:
                self.mu_dict["layer3"], self.sig_dict["layer3"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            # x = self.mixstyle_Sigma_dfed(x, mu_avg["layer3"], sig_avg["layer3"], Sigma_mu_avg["layer3"], Sigma_sig_avg["layer3"])
            x = self.mixstyle_Sigma_dfed(x, mus["layer3"], sigs["layer3"], Sigma_mus["layer3"], Sigma_sigs["layer3"])
        return self.layer4(x)

    # DDG-edited
    def forward(self, x, mu_avg=None, sig_avg=None, Sigma_mu_avg=None, Sigma_sig_avg=None, style_step=False):
        # DDG-edited
        f = self.featuremaps(x, mu_avg, sig_avg, Sigma_mu_avg, Sigma_sig_avg, style_step)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)

    # Author-added
    def extract_average_statistics(self):
        mu_dict_avg, sig_dict_avg = {}, {}
        Sigma_mu_dict_avg, Sigma_sig_dict_avg = {}, {}

        for layer in self.ms_layers:
            mu_dict_avg[layer] = self.mu_dict[layer].mean(dim=0, keepdim=True)
            sig_dict_avg[layer] = self.sig_dict[layer].mean(dim=0, keepdim=True)
            Sigma_mu_dict_avg[layer] = (self.mu_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
            Sigma_sig_dict_avg[layer] = (self.sig_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()

        return mu_dict_avg, sig_dict_avg, Sigma_mu_dict_avg, Sigma_sig_dict_avg


# Author-added (from DSU paper)
class UResNet(Backbone):

    def __init__(
        self,
        block,
        layers,
        pertubration=None,
        uncertainty=0.0,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # DDG-edited
        self.pertubration = pertubration(p=uncertainty) if pertubration else nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

        # DDG-added
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = False, False, False
        self.style_layer_num = 6

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.pertubration(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.pertubration(x)
        x = self.layer1(x)
        x = self.pertubration(x)
        x = self.layer2(x)
        x = self.pertubration(x)
        x = self.layer3(x)
        x = self.pertubration(x)
        x = self.layer4(x)
        x = self.pertubration(x)

        return x

    def forward(self, x, shift=False, mu_shift_dict=None, sig_shift_dict=None, label=None):
        if label == None:
            f = self.featuremaps(x)
        else:
            f = self.featuremaps(x, label)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


# DDG-edited (from DSU paper)
class UResNet_dfed(Backbone):

    def __init__(
        self,
        block,
        layers,
        pertubration_dfed=None,
        uncertainty=0.0,
        dsu_layers=[],
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # DDG-edited
        self.pertubration_dfed = pertubration_dfed(p=uncertainty) if pertubration_dfed else nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

        # DDG-added
        self.dsu_layers = dsu_layers
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = True, False, False
        self.mu_dict = {layer : None for layer in dsu_layers}
        self.std_dict = {layer : None for layer in dsu_layers}
        self.style_layer_num = len(dsu_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, mu_avg, std_avg, style_step=False):
        # DDG-added
        if mu_avg == None and std_avg == None:
            mu_avg = {layer : None for layer in self.dsu_layers}
            std_avg = {layer : None for layer in self.dsu_layers}

        x = self.conv1(x)
        if style_step:
            self.mu_dict["layer1"], self.std_dict["layer1"], _, _ = self.pertubration_dfed.extract_statistics(x)
        x = self.pertubration_dfed(x, mu_avg['layer1'], std_avg['layer1'])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if style_step:
            self.mu_dict["layer2"], self.std_dict["layer2"], _, _ = self.pertubration_dfed.extract_statistics(x)
        x = self.pertubration_dfed(x, mu_avg['layer2'], std_avg['layer2'])
        x = self.layer1(x)
        if style_step:
            self.mu_dict["layer3"], self.std_dict["layer3"], _, _ = self.pertubration_dfed.extract_statistics(x)
        x = self.pertubration_dfed(x, mu_avg['layer3'], std_avg['layer3'])
        x = self.layer2(x)
        if style_step:
            self.mu_dict["layer4"], self.std_dict["layer4"], _, _ = self.pertubration_dfed.extract_statistics(x)
        x = self.pertubration_dfed(x, mu_avg['layer4'], std_avg['layer4'])
        x = self.layer3(x)
        if style_step:
            self.mu_dict["layer5"], self.std_dict["layer5"], _, _ = self.pertubration_dfed.extract_statistics(x)
        x = self.pertubration_dfed(x, mu_avg['layer5'], std_avg['layer5'])
        x = self.layer4(x)
        if style_step:
            self.mu_dict["layer6"], self.std_dict["layer6"], _, _ = self.pertubration_dfed.extract_statistics(x)
        x = self.pertubration_dfed(x, mu_avg['layer6'], std_avg['layer6'])

        return x

    def forward(self, x, mu_avg=None, std_avg=None, style_step=False, label=None):
        if label == None:
            f = self.featuremaps(x, mu_avg, std_avg, style_step)
        else:
            f = self.featuremaps(x, mu_avg, std_avg, style_step, label)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)

    # Author-added
    def extract_average_statistics(self):
        mu_dict_avg, std_dict_avg = {}, {}
        Sigma_mu_dict_avg, Sigma_std_dict_avg = {}, {}

        for layer in self.dsu_layers:
            mu_dict_avg[layer] = self.mu_dict[layer].mean(dim=0, keepdim=True)
            std_dict_avg[layer] = self.std_dict[layer].mean(dim=0, keepdim=True)
            Sigma_mu_dict_avg[layer] = (self.mu_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
            Sigma_std_dict_avg[layer] = (self.std_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()

        return mu_dict_avg, std_dict_avg, Sigma_mu_dict_avg, Sigma_std_dict_avg

    # DDG-added
    def extract_full_statistics(self):
        return self.mu_dict, self.std_dict


# DDG-edited (from DSU paper)
class UResNet_Sigma_dfed(Backbone):

    def __init__(
        self,
        block,
        layers,
        pertubration_Sigma_dfed=None,
        uncertainty=0.0,
        dsu_layers=[],
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # DDG-edited
        self.pertubration_Sigma_dfed = pertubration_Sigma_dfed(p=uncertainty) if pertubration_Sigma_dfed else nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

        # DDG-added
        self.dsu_layers = dsu_layers
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = True, True, False
        self.mu_dict = {layer : None for layer in dsu_layers}
        self.std_dict = {layer : None for layer in dsu_layers}
        self.style_layer_num = len(dsu_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, mu_avg, std_avg, Sigma_mu_avg, Sigma_std_avg, style_step):
        # DDG-added
        if mu_avg == None and std_avg == None:
            # mu_avg = {layer : None for layer in self.dsu_layers}
            # std_avg = {layer : None for layer in self.dsu_layers}
            # Sigma_mu_avg = {layer : None for layer in self.dsu_layers}
            # Sigma_std_avg = {layer : None for layer in self.dsu_layers}
            pass

        mus, stds, Sigma_mus, Sigma_stds = {}, {}, {}, {}
        for layer in self.dsu_layers:
            mus[layer], stds[layer], Sigma_mus[layer], Sigma_stds[layer] = None, None, None, None
            if mu_avg != None:
                mus[layer], stds[layer], Sigma_mus[layer], Sigma_stds[layer] = {}, {}, {}, {}
                for neighbor_name in mu_avg:
                    mus[layer][neighbor_name] = mu_avg[neighbor_name][layer]
                    stds[layer][neighbor_name] = std_avg[neighbor_name][layer]
                    Sigma_mus[layer][neighbor_name] = Sigma_mu_avg[neighbor_name][layer]
                    Sigma_stds[layer][neighbor_name] = Sigma_std_avg[neighbor_name][layer]

        x = self.conv1(x)
        if style_step:
            self.mu_dict["layer1"], self.std_dict["layer1"], _, _ = self.pertubration_Sigma_dfed.extract_statistics(x)
        # x = self.pertubration_Sigma_dfed(x, mu_avg['layer1'], std_avg['layer1'], Sigma_mu_avg['layer1'], Sigma_std_avg['layer1'])
        # x = self.pertubration_Sigma_dfed(x, mus['layer1'], stds['layer1'], Sigma_mus['layer1'], Sigma_stds['layer1'])
        x = self.pertubration_Sigma_dfed(x, None, None, None, None)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if style_step:
            self.mu_dict["layer2"], self.std_dict["layer2"], _, _ = self.pertubration_Sigma_dfed.extract_statistics(x)
        # x = self.pertubration_Sigma_dfed(x, mu_avg['layer2'], std_avg['layer2'], Sigma_mu_avg['layer2'], Sigma_std_avg['layer2'])
        # x = self.pertubration_Sigma_dfed(x, mus['layer2'], stds['layer2'], Sigma_mus['layer2'], Sigma_stds['layer2'])
        x = self.pertubration_Sigma_dfed(x, None, None, None, None)
        x = self.layer1(x)
        if style_step:
            self.mu_dict["layer3"], self.std_dict["layer3"], _, _ = self.pertubration_Sigma_dfed.extract_statistics(x)
        # x = self.pertubration_Sigma_dfed(x, mu_avg['layer3'], std_avg['layer3'], Sigma_mu_avg['layer3'], Sigma_mu_avg['layer3'])
        x = self.pertubration_Sigma_dfed(x, mus['layer3'], stds['layer3'], Sigma_mus['layer3'], Sigma_stds['layer3'])
        x = self.layer2(x)
        if style_step:
            self.mu_dict["layer4"], self.std_dict["layer4"], _, _ = self.pertubration_Sigma_dfed.extract_statistics(x)
        # x = self.pertubration_Sigma_dfed(x, mu_avg['layer4'], std_avg['layer4'], Sigma_mu_avg['layer4'], Sigma_mu_avg['layer4'])
        x = self.pertubration_Sigma_dfed(x, mus['layer4'], stds['layer4'], Sigma_mus['layer4'], Sigma_stds['layer4'])
        x = self.layer3(x)
        if style_step:
            self.mu_dict["layer5"], self.std_dict["layer5"], _, _ = self.pertubration_Sigma_dfed.extract_statistics(x)
        # x = self.pertubration_Sigma_dfed(x, mu_avg['layer5'], std_avg['layer5'], Sigma_mu_avg['layer5'], Sigma_mu_avg['layer5'])
        x = self.pertubration_Sigma_dfed(x, mus['layer5'], stds['layer5'], Sigma_mus['layer5'], Sigma_stds['layer5'])
        x = self.layer4(x)
        if style_step:
            self.mu_dict["layer6"], self.std_dict["layer6"], _, _ = self.pertubration_Sigma_dfed.extract_statistics(x)
        # x = self.pertubration_Sigma_dfed(x, mu_avg['layer6'], std_avg['layer6'], Sigma_mu_avg['layer6'], Sigma_mu_avg['layer6'])
        # x = self.pertubration_Sigma_dfed(x, mus['layer6'], stds['layer6'], Sigma_mus['layer6'], Sigma_stds['layer6'])
        x = self.pertubration_Sigma_dfed(x, None, None, None, None)

        return x

    def forward(self, x, mu_avg=None, std_avg=None, Sigma_mu_avg=None, Sigma_std_avg=None, style_step=False, label=None):
        if label == None:
            f = self.featuremaps(x, mu_avg, std_avg, Sigma_mu_avg, Sigma_std_avg, style_step)
        else:
            f = self.featuremaps(x, mu_avg, std_avg, Sigma_mu_avg, Sigma_std_avg, style_step, label)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)

    # Author-added
    def extract_average_statistics(self):
        mu_dict_avg, std_dict_avg = {}, {}
        Sigma_mu_dict_avg, Sigma_std_dict_avg = {}, {}

        for layer in self.dsu_layers:
            mu_dict_avg[layer] = self.mu_dict[layer].mean(dim=0, keepdim=True)
            std_dict_avg[layer] = self.std_dict[layer].mean(dim=0, keepdim=True)
            Sigma_mu_dict_avg[layer] = (self.mu_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
            Sigma_std_dict_avg[layer] = (self.std_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()

        return mu_dict_avg, std_dict_avg, Sigma_mu_dict_avg, Sigma_std_dict_avg


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


class ResNet_Sigma_dfed_alternating(Backbone):

    def __init__(
        self,
        block,
        layers,
        ms_class=None,
        ms_layers=[],
        ms_p=0.5,
        ms_q=1,
        ms_a=0.1,
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.mixstyle_Sigma_dfed = None
        if ms_layers:
            self.mixstyle_Sigma_dfed = ms_class(p=ms_p, q=ms_q, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ["layer1", "layer2", "layer3"]
            print(
                f"Insert {self.mixstyle_Sigma_dfed.__class__.__name__} after {ms_layers}"
            )
        self.ms_layers = ms_layers

        self._init_params()

        # DDG-added
        self.style_sharing, self.style_Sigma_sharing, self.alternate_layers = True, True, True
        self.mu_dict = {layer : None for layer in ms_layers}
        self.sig_dict = {layer : None for layer in ms_layers}
        self.Sigma_mu_dict = {layer : None for layer in ms_layers}
        self.Sigma_sig_dict = {layer : None for layer in ms_layers}
        self.style_layer_num = len(ms_layers)
        self.current_layer_idx = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # DDG-edited
    def featuremaps(self, x, mu_avg=None, sig_avg=None, Sigma_mu_avg=None, Sigma_sig_avg=None, style_step=False):
        # DDG-added
        if mu_avg == None and sig_avg == None and Sigma_mu_avg == None and Sigma_sig_avg == None:
            mu_avg = {layer : None for layer in self.ms_layers}
            sig_avg = {layer : None for layer in self.ms_layers}
            Sigma_mu_avg = {layer : None for layer in self.ms_layers}
            Sigma_sig_avg = {layer : None for layer in self.ms_layers}
        for layer in self.ms_layers:
            if layer != self.ms_layers[self.current_layer_idx]:
                mu_avg[layer], sig_avg[layer], Sigma_mu_avg[layer], Sigma_sig_avg[layer] = None, None, None, None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.ms_layers[self.current_layer_idx] in ["layer1"]:
            # DDG-edited
            if style_step:
                self.mu_dict["layer1"], self.sig_dict["layer1"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            x = self.mixstyle_Sigma_dfed(x, mu_avg["layer1"], sig_avg["layer1"], Sigma_mu_avg["layer1"], Sigma_sig_avg["layer1"])
        x = self.layer2(x)
        if self.ms_layers[self.current_layer_idx] in ["layer1", "layer2"]:
            # DDG-edited
            if style_step:
                self.mu_dict["layer2"], self.sig_dict["layer2"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            x = self.mixstyle_Sigma_dfed(x, mu_avg["layer2"], sig_avg["layer2"], Sigma_mu_avg["layer2"], Sigma_sig_avg["layer2"])
        x = self.layer3(x)
        if self.ms_layers[self.current_layer_idx] in ["layer1", "layer2", "layer3"]:
            # DDG-edited
            if style_step:
                self.mu_dict["layer3"], self.sig_dict["layer3"] = self.mixstyle_Sigma_dfed.extract_statistics(x)
            x = self.mixstyle_Sigma_dfed(x, mu_avg["layer3"], sig_avg["layer3"], Sigma_mu_avg["layer3"], Sigma_sig_avg["layer3"])
        return self.layer4(x)

    # DDG-edited
    def forward(self, x, mu_avg=None, sig_avg=None, Sigma_mu_avg=None, Sigma_sig_avg=None, alternate_layer=False, style_step=False):
        # DDG-edited
        f = self.featuremaps(x, mu_avg, sig_avg, Sigma_mu_avg, Sigma_sig_avg, style_step)
        # DDG-added
        if alternate_layer:
            self.current_layer_idx = (self.current_layer_idx + 1) % self.style_layer_num
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)

    # Author-added
    def extract_average_statistics(self):
        mu_dict_avg, sig_dict_avg = {}, {}
        Sigma_mu_dict_avg, Sigma_sig_dict_avg = {}, {}

        for layer in self.ms_layers:
            if self.mu_dict[layer] != None and self.sig_dict[layer] != None :
                mu_dict_avg[layer] = self.mu_dict[layer].mean(dim=0, keepdim=True)
                sig_dict_avg[layer] = self.sig_dict[layer].mean(dim=0, keepdim=True)
                Sigma_mu_dict_avg[layer] = (self.mu_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
                Sigma_sig_dict_avg[layer] = (self.sig_dict[layer].var(dim=0, keepdim=True) + 1e-6).sqrt()
            else:
                mu_dict_avg[layer], sig_dict_avg[layer], Sigma_mu_dict_avg[layer], Sigma_sig_dict_avg[layer] = None, None, None, None

        return mu_dict_avg, sig_dict_avg, Sigma_mu_dict_avg, Sigma_sig_dict_avg


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
uresnet18: block=BasicBlock, layers=[2, 2, 2, 2]    # DDG-added
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


@BACKBONE_REGISTRY.register()
def resnet18(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def uresnet18(pretrained=True, uncertainty=0.5, **kwargs):
    from dassl.modeling.ops import DistributionUncertainty

    model = UResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration=DistributionUncertainty, uncertainty=uncertainty)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def uresnet18_dfed(pretrained=True, uncertainty=0.5, **kwargs):
    from dassl.modeling.ops import DistributionUncertainty_dfed

    model = UResNet_dfed(block=BasicBlock, layers=[2, 2, 2, 2],
                         pertubration_dfed=DistributionUncertainty_dfed, uncertainty=uncertainty,
                         dsu_layers=["layer1", "layer2", "layer3", "layer4", "layer5", "layer6"])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def uresnet18_Sigma_dfed(pretrained=True, uncertainty=0.5, **kwargs):
    from dassl.modeling.ops import DistributionUncertainty_Sigma_dfed

    model = UResNet_Sigma_dfed(block=BasicBlock, layers=[2, 2, 2, 2],
                               pertubration_Sigma_dfed=DistributionUncertainty_Sigma_dfed, uncertainty=uncertainty,
                               dsu_layers=["layer1", "layer2", "layer3", "layer4", "layer5", "layer6"])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet34(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet34"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet152(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet152"])

    return model


"""
Residual networks with mixstyle
"""


@BACKBONE_REGISTRY.register()
def resnet18_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_Stable_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import StyleExplore

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=StyleExplore,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l123_dfed(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle_dfed

    model = ResNet_dfed(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle_dfed,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l123_Sigma_dfed(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle_Sigma_dfed

    model = ResNet_Sigma_dfed(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle_Sigma_dfed,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l123_Sigma_dfed_alternating(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle_Sigma_dfed

    model = ResNet_Sigma_dfed_alternating(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle_Sigma_dfed,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_Stable_l123_Sigma_dfed(pretrained=True, **kwargs):
    from dassl.modeling.ops import StyleExplore_Sigma_dfed

    model = ResNet_Sigma_dfed(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=StyleExplore_Sigma_dfed,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model



@BACKBONE_REGISTRY.register()
def resnet18_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_ms_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l1_dfed(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle_dfed

    model = ResNet_dfed(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle_dfed,
        ms_layers=["layer1"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l1_Sigma_dfed(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle_Sigma_dfed

    model = ResNet_Sigma_dfed(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle_Sigma_dfed,
        ms_layers=["layer1"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# SOTS-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l2(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer2"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


# SOTS-added
@BACKBONE_REGISTRY.register()
def resnet18_ms_l3(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=MixStyle,
        ms_layers=["layer3"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def resnet50_Stable_l123_Sigma_dfed(pretrained=True, **kwargs):
    from dassl.modeling.ops import StyleExplore_Sigma_dfed

    model = ResNet_Sigma_dfed(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=StyleExplore_Sigma_dfed,
        ms_layers=["layer1", "layer2", "layer3"],
        ms_p=0.5
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


# DDG-added
@BACKBONE_REGISTRY.register()
def uresnet50(pretrained=True, uncertainty=0.5, **kwargs):
    from dassl.modeling.ops import DistributionUncertainty

    model = UResNet(block=Bottleneck, layers=[3, 4, 6, 3],
                    pertubration=DistributionUncertainty, uncertainty=uncertainty)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_ms_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=MixStyle,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


"""
Residual networks with efdmix
"""


@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_efdmix_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_class=EFDMix,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_efdmix_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_class=EFDMix,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_efdmix_l123(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2", "layer3"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=EFDMix,
        ms_layers=["layer1", "layer2"],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model


@BACKBONE_REGISTRY.register()
def resnet101_efdmix_l1(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        ms_class=EFDMix,
        ms_layers=["layer1"]
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet101"])

    return model
