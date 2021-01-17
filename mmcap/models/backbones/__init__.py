#from .darknet import Darknet
#from .detectors_resnet import DetectoRS_ResNet
#from .detectors_resnext import DetectoRS_ResNeXt
#from .hourglass import HourglassNet
#from .hrnet import HRNet
#from .regnet import RegNet
#from .resnet import ResNet, ResNetV1d
#from .resnext import ResNeXt
#from .ssd_vgg import SSDVGG
from .res2net import Res2Net
#from .cbnet import CBNet

__all__ = [
     'Res2Net',
#     'CBNet'
#    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
#    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
]
