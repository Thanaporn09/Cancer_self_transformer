# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
from .lenet import LeNet5
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .regnet import RegNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
#from .swin_transformer import SwinTransformer
from .timm_backbone import TIMMBackbone
from .tnt import TNT
from .vgg import VGG
from .vision_transformer import VisionTransformer
from .mit import MixVisionTransformer
from .vit import VisionTransformer_seg
#from .T2T_vit import T2T_ViT
from .CPT import ConvolutionalPyramidVisionTransformer, ConvolutionalPyramidVisionTransformerV2
from .PVT import PyramidVisionTransformer, PyramidVisionTransformerV2
from .swin import SwinTransformer
from .PiT import PoolingTransformer
from .CVT import ConvolutionalVisionTransformer
from .conformer import Conformer
from .deit import DistilledVisionTransformer
from .t2t_vit import T2T_ViT

__all__ = [
    'LeNet5', 'AlexNet', 'VGG', 'RegNet', 'ResNet', 'ResNeXt', 'ResNetV1d',
    'ResNeSt', 'ResNet_CIFAR', 'SEResNet', 'SEResNeXt', 'ShuffleNetV1',
    'ShuffleNetV2', 'MobileNetV2', 'MobileNetV3', 'VisionTransformer',
    'SwinTransformer', 'TNT', 'TIMMBackbone','MixVisionTransformer','T2T_ViT','PyramidVisionTransformer', 'PyramidVisionTransformerV2',
    'ConvolutionalPyramidVisionTransformer', 'ConvolutionalPyramidVisionTransformerV2','PoolingTransformer','ConvolutionalVisionTransformer','Conformer','DistilledVisionTransformer','VisionTransformer_seg'

]
