from resnet import resnet18
from generator import Generator
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from utils import concat_all_gather
from functools import partial

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, channel_per_group = 16):
        if num_channels <= channel_per_group:
            num_groups = 1
        else:
            num_groups = num_channels // channel_per_group
        super().__init__(num_groups, num_channels)

class Discriminator(nn.Module):
    def __init__(self, args, num_of_cat):
        super().__init__()
        latent_dim = 512
        self.encoder = resnet18(norm_layer = partial(GroupNorm, channel_per_group = args.gn_channel_per_group),  act = nn.ELU)
        self.fc_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ELU(), nn.Linear(latent_dim, args.project_dim))
        self.linear_probing = nn.Linear(latent_dim, num_of_cat)
        self.K = args.cov_queue_len
        # create the queue
        self.register_buffer("queue", F.normalize(torch.randn(self.K, args.project_dim), dim = -1))
        self.register_buffer("bg_queue", F.normalize(torch.randn(self.K, latent_dim), dim = -1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, feat, bg_feat):
        # gather keys before updating queue
        feat = concat_all_gather(feat)
        bg_feat = concat_all_gather(bg_feat)
        batch_size = feat.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        self.queue[ptr:ptr + batch_size,:] = feat
        self.bg_queue[ptr:ptr + batch_size,:] = bg_feat
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, img):
        backbone_feat = self.encoder(img)
        embed = self.fc_head(backbone_feat)
        norm_embed = F.normalize(embed, dim = -1)
        return norm_embed, embed, self.linear_probing(backbone_feat.detach()), F.normalize(backbone_feat, dim = -1)

class Generator_model(nn.Module):
    def __init__(self, latent_dim = 128, sn_scale = 1.0):
        super().__init__()
        self.G = Generator(G_ch=128, G_depth=2, dim_z=latent_dim, bottom_width=4, resolution=32,
                     G_kernel_size=3, G_attn='64', n_classes=1,
                     num_G_SVs=1, num_G_SV_itrs=1,
                     G_shared=True, shared_dim=0, hier=True,
                     cross_replica=False, mybn=False,
                     G_activation=nn.ReLU(inplace=False),
                     G_lr=0.0001, G_B1=0.0, G_B2=0.999, adam_eps=1e-06,
                     BN_eps=1e-5, SN_eps=1e-06, G_mixed_precision=False, G_fp16=False,
                     G_init='kaiming', skip_init=False, no_optim=True,
                     G_param='SN', norm_style='bn', scale = 1)

    def forward(self, z):
        img = self.G(z)
        return img
