# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, constant_init, kaiming_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class VisionTransformerClsHead(ClsHead):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer. Only
            available during pre-training. Default None.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defalut Tanh.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=None,
                 act_cfg=dict(type='Tanh'),
                 *args,
                 **kwargs):
        super(VisionTransformerClsHead, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = nn.Sequential(OrderedDict(layers))

    def init_weights(self):
        super(VisionTransformerClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            kaiming_init(
                self.layers.pre_logits, mode='fan_in', nonlinearity='linear')
        constant_init(self.layers.head, 0)

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[tuple[tensor, tensor]]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. Every item should be a tuple which
                includes patch token and cls token. The cls token will be used
                to classify and the shape of it should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.layers(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
#        print('x:',x.shape)
#        print('gt_label:',gt_label.shape)
        cls_score = self.layers(x)
#        print(cls_score.shape)
        losses = self.loss(cls_score, gt_label)
        return losses
