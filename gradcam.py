#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        index = torch.argmax(self.logits, 1)
        output = self.logits
        return index, output

    def backward(self, ids, output):
        """
        Class-specific backpropagation
        """
        one_hot = torch.nn.functional.one_hot(ids, num_classes=output.shape[1])
        one_hot = torch.sum(one_hot.to(self.device) * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                grad_out[0].requires_grad_()
                self.grad_pool[key] = grad_out[0]

            return backward_hook

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = weights * fmaps
        gcam[torch.abs(gcam) < 1e-20] = 0.0
        B, C, H, W = gcam.shape
        norms = gcam.reshape(B, C, -1)
        norms = torch.linalg.norm(norms, ord=2, dim=-1, keepdim=True)

        norms = norms.unsqueeze(-1).repeat(1, 1, H, W)
        gcam = gcam / norms
        gcam[torch.isnan(gcam)] = 0.0
        gcam[torch.isinf(gcam)] = 0.0

        gcam = [gcam]

        return gcam

    def clear(self):
        for handle in self.handlers:
            handle.remove()
        del self.model, self.image_shape, self.logits, self.probs, self.fmap_pool, self.grad_pool
