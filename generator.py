import numpy as np
import math,pdb
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
#import layers as new_layers
import biggan_layers as layers
#import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d
from torch.nn import LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer
from torch.nn.parameter import UninitializedParameter
# BigGAN-deep: uses a different resblock and pattern


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
class LazyLayerNorm(LazyModuleMixin, LayerNorm):
    """
    A `LayerNorm` with lazy initialization.
    See `LayerNorm` for details:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
    Parameters
    ----------
    - eps : float, default 1e-5
        a value added to the denominator for numerical stability.
    - elementwise_affine : bool, default True
        a boolean value that when set to True, this module has learnable per-element
        affine parameters initialized to ones (for weights) and zeros (for biases).
    Examples
    --------
    >>> input = torch.randn(20, 5, 10, 10)
    >>> # With Learnable Parameters
    >>> m = LazyLayerNorm()
    >>> # Without Learnable Parameters
    >>> m = LazyLayerNorm(elementwise_affine=False)
    >>> m
    LazyLayerNorm((0,), eps=1e-05, elementwise_affine=False)
    >>> output = m(input)
    >>> output.size()
    torch.Size([20, 5, 10, 10])
    >>> m
    LayerNorm((5, 10, 10), eps=1e-05, elementwise_affine=False)
    """

    cls_to_become = LayerNorm

    def __init__(self, latent_dim, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__(0, eps, elementwise_affine)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def initialize_parameters(self, input) -> None:
        self.normalized_shape = tuple(input.size()[1:])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape)
                self.bias.materialize(self.normalized_shape)
                torch.nn.init.ones_(self.weight)
                torch.nn.init.zeros_(self.bias)

# Channel ratio is the ratio of
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=layers.bn, activation=None,
               upsample=None, channel_ratio=4):
    super(GBlock, self).__init__()

    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.in_channels // channel_ratio
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                 kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(self.in_channels)
    self.bn2 = self.which_bn(self.hidden_channels)
    self.bn3 = self.which_bn(self.hidden_channels)
    self.bn4 = self.which_bn(self.hidden_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    # Project down to channel ratio
    h = self.conv1(self.activation(self.bn1(x, y)))
    # Apply next BN-ReLU
    h = self.activation(self.bn2(h, y))
    # Drop channels in x if necessary
    if self.in_channels != self.out_channels:
      x = x[:, :self.out_channels]
    # Upsample both h and x at this point
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    # 3x3 convs
    h = self.conv2(h)
    h = self.conv3(self.activation(self.bn3(h, y)))
    # Final 1x1 conv
    h = self.conv4(self.activation(self.bn4(h, y)))
    return h + x

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4,  2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

class Generator(nn.Module):
  def __init__(self, G_ch=64, G_depth=2, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn', scale = 1.0,
               **kwargs):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Number of resblocks per stage
    self.G_depth = G_depth
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim
    # Hierarchical latent space?
    self.hier = hier
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps, scale = scale, use_sn = True)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                          eps=self.SN_eps, scale = scale, use_sn = True)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear

    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.dim_z if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    #self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
    #                else layers.identity())
    self.shared = layers.identity()
    # First linear layer
    #self.linear = self.which_linear(self.dim_z + self.shared_dim, self.arch['in_channels'][0] * (self.bottom_width **2))
    self.linear = self.which_linear(self.dim_z, self.arch['in_channels'][0] * (self.bottom_width **2))
    self.firs_layer = self.which_linear(self.dim_z, self.dim_z)
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['in_channels'][index] if g_index==0 else self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] and g_index == (self.G_depth-1) else None))]
                       for g_index in range(self.G_depth)]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'kaiming':
          init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  # NOTE: The z vs y dichotomy here is for compatibility with not-y
  def forward(self, z):
    # If hierarchical, concatenate zs and ys
    z = self.firs_layer(z)
    if self.hier:
      #z = torch.cat([y, z], 1)
      y = z
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, y)

    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))
