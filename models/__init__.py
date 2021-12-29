__all__ = ['residual_unet','unet','unet++','init_weight','Attention_unet']
from .unet import unet
from .Attention_unet import attention_unet
import torch.nn as nn

def init_weight(m):
    if  isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
    if isinstance(m,nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')

