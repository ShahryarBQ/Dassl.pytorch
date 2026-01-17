import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from random import shuffle, choice

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

# DDG-added
import networkx as nx
from torch import linalg as LA


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

        # DDG-added
        self.style_layer_num = self.backbone.style_layer_num

    @property
    def fdim(self):
        return self._fdim

    # SOTS-edited
    def forward(self, x, return_feature=False, shift=False, mu_shift_dict=None, sig_shift_dict=None):
        # SOTS-edited
        if shift:
            f = self.backbone(x, shift, mu_shift_dict, sig_shift_dict)
        else:
            f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class SimpleNet_dfed(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

        # DDG-added
        self.style_sharing = self.backbone.style_sharing
        self.style_Sigma_sharing = self.backbone.style_Sigma_sharing
        self.alternate_layers = self.backbone.alternate_layers
        self.style_layer_num = self.backbone.style_layer_num

    @property
    def fdim(self):
        return self._fdim

    # DDG-edited
    def forward(self, x, return_feature=False, mu_avg=None, sig_avg=None, Sigma_mu_avg=None, Sigma_sig_avg=None, alternate_layer=False, style_step=False):
        # DDG-edited
        if self.style_sharing and self.style_Sigma_sharing and self.alternate_layers:
            f = self.backbone(x, mu_avg, sig_avg, Sigma_mu_avg, Sigma_sig_avg, alternate_layer, style_step)
        elif self.style_sharing and self.style_Sigma_sharing:
            f = self.backbone(x, mu_avg, sig_avg, Sigma_mu_avg, Sigma_sig_avg, style_step)
        elif self.style_sharing and self.alternate_layers:
            f = self.backbone(x, mu_avg, sig_avg, alternate_layer, style_step)
        elif self.style_sharing:
            f = self.backbone(x, mu_avg, sig_avg, style_step)
        else:
            f = self.backbone(x, style_step)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y

    # Author-added
    def extract_average_statistics(self):
        return self.backbone.extract_average_statistics()

    # DDG-added
    def extract_full_statistics(self):
        return self.backbone.extract_full_statistics()


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        # DDG-added
        self._mus = OrderedDict()
        self._sigs = OrderedDict()
        self._Sigma_mus = OrderedDict()
        self._Sigma_sigs = OrderedDict()
        self._style_clients = OrderedDict()
        self.gamma = None
        self._style_bank = OrderedDict()

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

        # DDG-added
        layer_num = model.style_layer_num
        self._mus[name] = {f"layer{i+1}" : None for i in range(layer_num)}
        self._sigs[name] = {f"layer{i+1}" : None for i in range(layer_num)}
        self._Sigma_mus[name] = {f"layer{i+1}" : None for i in range(layer_num)}
        self._Sigma_sigs[name] = {f"layer{i+1}" : None for i in range(layer_num)}
        self._style_clients[name] = {name}
        self.gamma = 0
        self._style_bank[name] = {"mus": {}, "sigs": {}, "Sigma_mus": {}, "Sigma_sigs": {}}

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        return
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    # Decentralized (Inspired by the StableFDG paper)
    def model_backward_and_update_dfed(self, losses_dfed, names=None):
        self.model_zero_grad(names)
        self.models_backward_dfed(losses_dfed)
        self.model_update(names)
        # for _ in range(self.num_clients * self.num_clients):
        self.models_aggregate_dfed(names)

    # DDG-added
    def styles_backward_and_update_dfed(self, names=None):
        self.styles_update_dfed(names)
        hops = 3
        for _ in range(hops):
            self.styles_aggregate_dfed(names)

        # self.styles_Sigma_update_dfed(names)
        # # for _ in range(self.num_clients * self.num_clients):
        # self.styles_Sigma_aggregate_dfed(names)

    # Decentralized (Inspired by the StableFDG paper)
    def models_backward_dfed(self, losses_dfed):
        for loss in list(losses_dfed.values()):
            self.detect_anomaly(loss)
            loss.backward()

    # DDG-added
    def models_aggregate_dfed(self, names=None):
        names = self.get_model_names(names)
        local_models = {}
        for name in names:
            local_models[name] = deepcopy(self._models[name].state_dict())

        for name in names:
            for neighbor_name in self.graph_adj[name]:
                aggregation_weight = self.model_aggregation_weight(name, neighbor_name)

                for layer in local_models[name]:
                    if 'bn' not in layer:
                        with torch.no_grad():
                            local_models[name][layer] = torch.add(local_models[name][layer], torch.mul(aggregation_weight, self._models[neighbor_name].state_dict()[layer] - self._models[name].state_dict()[layer]))

        for name in names:
            self._models[name].load_state_dict(local_models[name])

    # DDG-added
    def styles_update_dfed(self, names):
        names = self.get_model_names(names)
        if not self._models[names[0]].style_sharing:
            return

        for name in names:
            self._mus[name], self._sigs[name], self._Sigma_mus[name], self._Sigma_sigs[name] = self._models[name].extract_average_statistics()
            # self._mus[name], self._sigs[name] = self._models[name].extract_full_statistics()

            for key in self._style_bank[name]:
                self._style_bank[name][key] = {}
            self._style_bank[name]["mus"][name] = self._mus[name]
            self._style_bank[name]["sigs"][name] = self._sigs[name]
            self._style_bank[name]["Sigma_mus"][name] = self._Sigma_mus[name]
            self._style_bank[name]["Sigma_sigs"][name] = self._Sigma_sigs[name]

    # DDG-added
    def styles_Sigma_update_dfed(self, names):
        names = self.get_model_names(names)
        if not self._models[names[0]].style_sharing or not self._models[names[0]].style_Sigma_sharing:
            return

        local_mus, local_sigs, local_Sigma_mus, local_Sigma_sigs = {}, {}, {}, {}
        for name in names:
            local_mus[name], local_sigs[name], local_Sigma_mus[name], local_Sigma_sigs[name] = self._models[name].extract_average_statistics()

            for layer in local_mus[name]:
                # self._Sigma_mus[name][layer] = torch.add(torch.pow(local_mus[name][layer] - self._mus[name][layer], 2).mean(dim=0, keepdim=True), 1e-6).sqrt()
                # self._Sigma_sigs[name][layer] = torch.add(torch.pow(local_mus[name][layer] - self._mus[name][layer], 2).mean(dim=0, keepdim=True), 1e-6).sqrt()

                continue

    # DDG-added
    def styles_aggregate_dfed(self, names=None):
        names = self.get_model_names(names)
        if not self._models[names[0]].style_sharing:
            return

        tmp_mus, tmp_sigs = deepcopy(self._mus), deepcopy(self._sigs)
        tmp_Sigma_mus, tmp_Sigma_sigs = deepcopy(self._Sigma_mus), deepcopy(self._Sigma_sigs)
        for name in names:
            # 3) Choose one of neighbors
            # neighbor_choice = choice(self.graph_adj[name])

            # 4) Popoulate the style bank
            for neighbor_name in self.graph_adj[name]:
                for avail_name in self._style_bank[neighbor_name]["mus"]:
                    self._style_bank[name]["mus"][avail_name] = self._style_bank[neighbor_name]["mus"][avail_name]
                    self._style_bank[name]["sigs"][avail_name] = self._style_bank[neighbor_name]["sigs"][avail_name]
                    self._style_bank[name]["Sigma_mus"][avail_name] = self._style_bank[neighbor_name]["Sigma_mus"][avail_name]
                    self._style_bank[name]["Sigma_sigs"][avail_name] = self._style_bank[neighbor_name]["Sigma_sigs"][avail_name]

            for layer in tmp_mus[name]:
                # self._mus[name][layer] = tmp_mus[neighbor_choice][layer]
                # self._sigs[name][layer] = tmp_sigs[neighbor_choice][layer]
                # self._Sigma_mus[name][layer] = tmp_Sigma_mus[neighbor_choice][layer]
                # self._Sigma_sigs[name][layer] = tmp_Sigma_sigs[neighbor_choice][layer]

                for neighbor_name in self.graph_adj[name]:
                    aggregation_weight = self.model_aggregation_weight(name, neighbor_name)

                    # # 1) Learn the global average
                    # self._mus[name][layer] = torch.add(self._mus[name][layer], torch.mul(aggregation_weight, tmp_mus[neighbor_name][layer] - tmp_mus[name][layer]))
                    # self._sigs[name][layer] = torch.add(self._sigs[name][layer], torch.mul(aggregation_weight, tmp_sigs[neighbor_name][layer] - tmp_sigs[name][layer]))

                    # # 2) Learn the average of neighbors
                    # self._mus[name][layer] = torch.add(self._mus[name][layer], torch.mul(1 / len(self.graph_adj[name]), tmp_mus[neighbor_name][layer] - tmp_mus[name][layer]))
                    # self._sigs[name][layer] = torch.add(self._sigs[name][layer], torch.mul(1 / len(self.graph_adj[name]), tmp_sigs[neighbor_name][layer] - tmp_sigs[name][layer]))

                    continue

    # DDG-added
    def styles_Sigma_aggregate_dfed(self, names=None):
        names = self.get_model_names(names)
        if not self._models[names[0]].style_sharing or not self._models[names[0]].style_Sigma_sharing:
            return

        avg_mus, avg_sigs = deepcopy(self._mus), deepcopy(self._sigs)
        tmp_mus, tmp_sigs = deepcopy(self._mus), deepcopy(self._sigs)
        for name in names:
            for layer in self._Sigma_mus[name]:
                for neighbor_name in self.graph_adj[name]:
                    aggregation_weight = self.model_aggregation_weight(name, neighbor_name)

                    avg_mus[name][layer] =  torch.add(self._mus[name][layer], torch.mul(aggregation_weight, tmp_mus[neighbor_name][layer] - tmp_mus[name][layer]))
                    avg_sigs[name][layer] = torch.add(self._sigs[name][layer], torch.mul(aggregation_weight, tmp_sigs[neighbor_name][layer] - tmp_sigs[name][layer]))

        for name in names:
            for layer in self._Sigma_mus[name]:
                self._Sigma_mus[name][layer] = torch.pow(self._Sigma_mus[name][layer], 2) + torch.pow(self._mus[name][layer], 2)
                self._Sigma_sigs[name][layer] = torch.pow(self._Sigma_sigs[name][layer], 2) + torch.pow(self._sigs[name][layer], 2)
        tmp_Sigma_mus, tmp_Sigma_sigs = deepcopy(self._Sigma_mus), deepcopy(self._Sigma_sigs)

        for name in names:
            for layer in self._Sigma_mus[name]:
                for neighbor_name in self.graph_adj[name]:
                    aggregation_weight = self.model_aggregation_weight(name, neighbor_name)

                    # 1) Learn the global average
                    self._Sigma_mus[name][layer] = torch.add(self._Sigma_mus[name][layer], torch.mul(aggregation_weight, tmp_Sigma_mus[neighbor_name][layer] - tmp_Sigma_mus[name][layer]))
                    self._Sigma_sigs[name][layer] = torch.add(self._Sigma_sigs[name][layer], torch.mul(aggregation_weight, tmp_Sigma_sigs[neighbor_name][layer] - tmp_Sigma_sigs[name][layer]))

                    # # 2) Learn the average of neighbors
                    # self._Sigma_mus[name][layer] = torch.add(self._Sigma_mus[name][layer], torch.mul(1 / len(self.graph_adj[name]), tmp_Sigma_mus[neighbor_name][layer] - tmp_Sigma_mus[name][layer]))
                    # self._Sigma_sigs[name][layer] = torch.add(self._Sigma_sigs[name][layer], torch.mul(1 / len(self.graph_adj[name]), tmp_Sigma_sigs[neighbor_name][layer] - tmp_Sigma_sigs[name][layer]))

                    continue

                self._Sigma_mus[name][layer] = (self._Sigma_mus[name][layer] - torch.pow(avg_mus[name][layer], 2)).sqrt()
                self._Sigma_sigs[name][layer] = (self._Sigma_sigs[name][layer] - torch.pow(avg_sigs[name][layer], 2)).sqrt()

    # # DDG-added
    # def styles_aggregate_dfed(self, names=None):
    #     names = self.get_model_names(names)
    #     if not self._models[names[0]].style_sharing:
    #         return

    #     tmp_mus, tmp_sigs = deepcopy(self._mus), deepcopy(self._sigs)
    #     local_mus, local_sigs = {}, {}
    #     for name in names:
    #         local_mus[name], local_sigs[name] = self._models[name].extract_average_statistics()

    #     for name in names:
    #         for neighbor_name in self.graph_adj[name]:
    #             aggregation_weight = self.calculate_aggregation_weight(name, neighbor_name)
    #             lr = self._scheds[name].get_last_lr()

    #             for layer in local_mus[name]:
    #                 self._mus[name][layer] = torch.add(self._mus[name][layer][name][layer], torch.mul(aggregation_weight, tmp_mus[neighbor_name][layer] - tmp_mus[name][layer]))
    #                 self._sigs[name][layer] = torch.add(torch.pow(self._sigs[name][layer], 2), torch.mul(aggregation_weight, torch.pow(tmp_sigs[neighbor_name][layer], 2) - torch.pow(tmp_sigs[name][layer], 2)))
    #                 # self._Sigma_mus[name][layer] = torch.add(self._Sigma_mus[name][layer], torch.mul(aggregation_weight, local_Sigma_mus[neighbor_name][layer] - local_Sigma_mus[name][layer]))
    #                 # self._Sigma_sigs[name][layer] = torch.add(self._Sigma_sigs[name][layer], torch.mul(aggregation_weight, local_Sigma_sigs[neighbor_name][layer] - local_Sigma_sigs[name][layer]))

    #                 self._mus[name][layer] = torch.sub(self._mus[name][layer], torch.mul(lr, torch.matmul(torch.transpose(self.forward_gradients[name][layer].mean(dim=[0, 2, 3]), 0, 1), self._models[name][layer].grad)))
    #                 self._sigs[name][layer] = torch.sub(self._sigs[name][layer], torch.mul(lr, torch.cov(self.outputs[name][layer], self.forward_gradients[name][layer], dim=[2, 3])))
    #                 self._sigs[name][layer] = torch.add(self._sigs[name][layer], torch.mul(torch.pow(lr, 2), torch.mul(self.forward_gradients[name][layer].var(dim=[2, 3], keepdim=True), torch.pow(LA.vector_norm(self._models[name][layer].grad), 2))))
    #                 self._sigs[name][layer] = self._sigs[name][layer].sqrt()

    # DDG-added
    def get_degree(self, name):
        return len(self.graph_adj[name])

    # DDG-added
    def model_aggregation_weight(self, src_name, dst_name):
        return 1 / (1 + max(self.get_degree(src_name), self.get_degree(dst_name)))

    # DDG-added
    def generate_graph(self, names=None):
        names = self.get_model_names(names)
        while True:
            print(self.num_clients, self.graph_conn)
            graph = nx.random_geometric_graph(self.num_clients, self.graph_conn)
            # graph = nx.erdos_renyi_graph(self.num_clients, p=2/self.num_clients)
            # graph = nx.scale_free_graph(self.num_clients).to_undirected()
            if nx.is_k_edge_connected(graph, 1):
                break

        graph_adj = {names[i] : [names[idx] for idx in list(graph.adj[i])] for i in range(self.num_clients)}
        return graph_adj


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            # self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # SOTS-added
        mix = self.cfg.TRAINER.VANILLA2.MIX
        mu_shift_dict, sig_shift_dict = None, None
        if mix == "sots":
            mu_shift_dict, sig_shift_dict = self.model.extract_average_statistics()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # SOTS-edited
            output = self.model_inference(input, mix, mu_shift_dict, sig_shift_dict)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    # SOTS-edited
    def model_inference(self, input, mix="random", mu_shift_dict=None, sig_shift_dict=None):
        if mix in ["random", "crossdomain"]:
            return self.model(input)
        elif mix == "sots":
            return self.model(input, shift=True, mu_shift_dict=mu_shift_dict, sig_shift_dict=sig_shift_dict)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


# Decentralized (DDG-added)
class SimpleTrainer_dfed(TrainerBase):
    """A simple trainer class implementing generic functions slightly modified for Decentralized version."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        # DDG-edited
        self.cfg = cfg
        self.num_clients = cfg.NUM_CLIENTS
        self.graph_conn = cfg.GRAPH_CONN
        self.build_data_loaders_dfed()    # Should be before build_models_dfed()
        self.build_models_dfed()
        self.evaluators = [build_evaluator(cfg, lab2cname=self.lab2cname) for _ in range(self.num_clients)]
        self.best_result = -np.inf

        # Decentralized (Inspired by the StableFDG paper)
        self.graph_adj = self.generate_graph()
        self.outputs = OrderedDict()
        self.forward_gradients = OrderedDict()

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loaders_dfed(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, num_clients=self.num_clients)

        # DDG-edited
        self.train_loader_x_dfed = dm.train_loader_x_dfed
        self.train_loader_u_dfed = dm.train_loader_u_dfed  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_models_dfed(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        # DDG-edited
        print("Building model")
        model = SimpleNet_dfed(cfg, cfg.MODEL, self.num_classes)
        self.models = [deepcopy(model) for _ in range(self.num_clients)]
        if cfg.MODEL.INIT_WEIGHTS:
            for model in self.models:
                load_pretrained_weights(model, cfg.MODEL.INIT_WEIGHTS)
        for model in self.models:
            model.to(self.device)
        print(f"# params: {count_num_param(self.models[0]):,}")
        self.optims = [None for _ in range(self.num_clients)]
        self.scheds = [None for _ in range(self.num_clients)]
        for i in range(self.num_clients):
            self.optims[i] = build_optimizer(self.models[i], cfg.OPTIM)
            self.scheds[i] = build_lr_scheduler(self.optims[i], cfg.OPTIM)
            self.register_model(f"model_{i}", self.models[i], self.optims[i], self.scheds[i])

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            # for i in range(self.num_clients):
            #     self.models[i] = nn.DataParallel(self.models[i])

        # DDG-added
        self.train_loader_x_dfed = {f"model_{i}" : self.train_loader_x_dfed[i] for i in range(self.num_clients)}

        # # DDG-added
        # for name in self.train_loader_x_dfed:
        #     input, labels = self.parse_batch_train(next(iter(self.train_loader_x_dfed[name])))
        #     outputs = self._models[name](input)
        #     if self._models[name].style_sharing:
        #         self._mus[name], self._sigs[name], self._Sigma_mus[name], self._Sigma_sigs[name] = self._models[name].extract_average_statistics()

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        # DDG-added
        # self.test()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        # DDG-edited
        for evaluator in self.evaluators:
            evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        names = self.get_model_names()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # DDG-edited
            for i in range(self.num_clients):
                name = names[i]
                output = self.model_inference(input, self._models[name])
                self.evaluators[i].process(output, label)

        # DDG-edited
        results_dfed, results_avg = [], {}
        for evaluator in self.evaluators:
            results_dfed.append(evaluator.evaluate())
        for key in results_dfed[0]:
            results_avg[key] = sum([results_dfed[i][key] for i in range(self.num_clients)]) / self.num_clients

        # DDG-added (from evaluator.py)
        print(
            "======> final result\n"
            f"* accuracy: {results_avg['accuracy']:.1f}%\n"
            f"* error: {results_avg['error_rate']:.1f}%\n"
            f"* macro_f1: {results_avg['macro_f1']:.1f}%"
        )

        # DDG-edited
        for k, v in results_avg.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results_avg.values())[0]

    # DDG-edited
    def model_inference(self, input, model):
        return model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain


# Decentralized (Inspired by the StableFDG paper)
class TrainerX_dfed(SimpleTrainer_dfed):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        #DDG-edited
        names = self.get_model_names()
        losses_dfed = {names[i] : MetricMeter() for i in range(self.num_clients)}
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = min([len(self.train_loader_x_dfed[name]) for name in names])

        end = time.time()
        # DDG-edited
        for self.batch_idx, batch in enumerate(zip(*self.train_loader_x_dfed.values())):
            data_time.update(time.time() - end)
            loss_summaries = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            for name in names:
                losses_dfed[name].update(loss_summaries[name])

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                for name in names:
                    info += [f"{losses_dfed[name]}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for _name in names:
                for name, meter in losses_dfed[_name].meters.items():
                    self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
