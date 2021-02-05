from . classification import AbstractClassificationModel
from . classification import KerasResNet50V2
from . classification import KerasResNet50
from . classification import KerasMobileNetV2

from . efficientnet import KerasEfficientNetB0
from . efficientnet import KerasEfficientNetB1
from . efficientnet import KerasEfficientNetB2
from . efficientnet import KerasEfficientNetB3
from . efficientnet import KerasEfficientNetB4
from . efficientnet import KerasEfficientNetB5
from . efficientnet import KerasEfficientNetB6
from . efficientnet import KerasEfficientNetB7
from . efficientnet import EfficientNetB0
from . classification import SimpleClassificationModel


# from . efficientnet import KerasEfficientNetB1
# from . efficientnet import KerasEfficientNetB2

from . small_resnet_keras_contrib import KerasResNet18
from . small_resnet_keras_contrib import KerasResNet34
# from . resnet import ResNet18
# from . resnet import ResNet34
# from . resnet import ResNet50

from . autoencoder import SimpleAutoEncoderModel
from . autoencoder import SSIMAutoEncoderModel

from . representation import SimpleRepresentationModel
from . representation import add_projection_layers

__all__ = [
    'AbstractClassificationModel'
    'KerasResNet18',
    'KerasResNet34',
    'KerasResNet50',
    'KerasResNet50V2',
    'KerasMobileNetV2',
    'KerasEfficientNetB0',
    'KerasEfficientNetB1',
    'KerasEfficientNetB2',
    'KerasEfficientNetB3',
    'KerasEfficientNetB4',
    'KerasEfficientNetB5',
    'KerasEfficientNetB6',
    'KerasEfficientNetB7',
    'EfficientNetB0',
    'SimpleClassificationModel',
    'SimpleAutoEncoderModel',
    'SSIMAutoEncoderModel',
    'SimpleRepresentationModel',
    'add_projection_layers'
]

"""
    'ConvAutoEncoder',
    'BaseModel',
    'SimpleBaseModel',
    'ResNet18',
    'ResNet34'
    'ResNet50V2',

"""