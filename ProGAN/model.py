import torch
import torch.nn as nn
import torch.nn.functional as F


class WSConv2d(nn.Module):
  def __init__(self, 
               in_channels: int, 
               out_channels: int, 
               kernel_size: int=3, 
               stride: int=1, 
               padding: int=1, 
               gain: float=2):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.scale = (gain / (in_channels * kernel_size**2))**0.5
    self.bias = self.conv.bias
    self.conv.bias = None
    nn.init.normal_(self.conv.weight)
    nn.init.zeros_(self.bias)

  def forward(self, x):
    return self.conv(x * self.scale) + self.bias.reshape(1, self.bias.shape[0], 1, 1)

