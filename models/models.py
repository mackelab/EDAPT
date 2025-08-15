"""Model registry and imports."""

from .eegnet import EEGNetv4
from .shallow_convnet import ShallowConvNet
from .deep_convnet import DeepConvNet
from .atcnet import ATCNet

MODEL_CLASS_MAP = {
    "EEGNetv4": EEGNetv4,
    "ShallowConvNet": ShallowConvNet,
    "DeepConvNet": DeepConvNet,
    "ATCNet": ATCNet,
}

PREFIX_MAP = {
    "EEGNetv4": "eegnet_",
    "ShallowConvNet": "shallow_",
    "DeepConvNet": "deep_",
    "ATCNet": "atcnet_",
}