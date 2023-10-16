import argparse, itertools
import os, json
import random
import shutil
import time, glob, copy
import os
import time
import torch
import socket
import argparse
import subprocess
import math
from tqdm import tqdm
import warnings
import numpy as np
import torch, pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse, pdb
import numpy as np
from torch import autograd
from torch.optim import Adam, SGD, AdamW
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable
from model import Generator_model, Discriminator
from biggan_layers import MovingBatchNorm
from utils import AverageMeter, ProgressMeter, compute_adv_loss, compute_knn_loss
import builtins
import torchvision.utils as vutils
from kmeans import eval_kmeans

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--img_size', default=224, type=int,
                    help='img size')
parser.add_argument('--batch_iter', default=48, type=int,
                    help='img size')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', default=5, type=int,
                     help='print frequency (default: 10)')
parser.add_argument('--resume', action = 'store_true',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument("--project_dim", type=int, default=128)
# optimizer param
parser.add_argument("--data_type", type=str, choices=['cifar10', 'cifar100'])
parser.add_argument("--learning_rate", type=float, default=1e-3, help = 'learning rate')
parser.add_argument("--weight_decay", type=float, default=0, help = 'weight decay ratio for discriminator')
parser.add_argument("--momentum_model", type=float, default=0.99, help = 'momentum ratio of the moving average')
parser.add_argument("--KNN", type=int, default=1, help = 'number of nearest neighbours')
parser.add_argument("--gn_channel_per_group", type=int, default=16, help = 'number of channels of each group in group normalization')
parser.add_argument("--cov_queue_len", type=int, default=10240, help = 'lenght of memory bank')
# datamodule params
parser.add_argument("--data_path", type=str, default=".", help = 'folder path of the dataset')
parser.add_argument("--num_of_cat", type=int, default=10, help = 'number of categories in the dataset')
parser.add_argument("--batch_size", type=int, default=128, help = 'per-gpu batch size ')

args = parser.parse_args()

import random

from PIL import ImageFilter


EPS = 1e-20

def prob_gradients(model):
    norms = []
    max_grad = 0
    max_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            max_grad = max(max_grad, p.grad.norm())
            max_norm = max(max_norm, p.data.norm(2))
    return max_grad, max_norm

def main():
    import os
    #torch.backends.cudnn.benchmark=False
    cudnn.deterministic = True
    args = parser.parse_args()
    #assert args.batch_size % args.batch_iter == 0
    if not os.path.exists('visualize'):
        os.system('mkdir visualize')
    if not os.path.exists('checkpoint'):
        os.system('mkdir checkpoint')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args = copy.deepcopy(args)
    args.cos = True
    save_folder_path = 'checkpoint_gan/lr_{}_weight_decay_{}_data_{}_momentum_{}_KNN_{}'.format(
                        args.learning_rate, args.weight_decay,
                        args.data_type,
                        args.momentum_model,
                        args.KNN)

    args.save_folder_path = save_folder_path
    args.is_master = args.rank == 0

    model_G = Generator_model()
    model_G.cuda(args.gpu)
    moment_model_G = Generator_model()
    moment_model_G.cuda(args.gpu)
    model_D = Discriminator(args, num_of_cat = args.num_of_cat)
    model_D.cuda(args.gpu)
    moment_model_D = Discriminator(args, num_of_cat =  args.num_of_cat)
    moment_model_D.cuda(args.gpu)

    model_D = torch.nn.parallel.DistributedDataParallel(model_D, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    moment_model_D = torch.nn.DataParallel(moment_model_D, device_ids=[args.gpu])
    model_G = torch.nn.parallel.DistributedDataParallel(model_G, device_ids=[args.gpu], find_unused_parameters=True,
    broadcast_buffers=False)
    moment_model_G = torch.nn.DataParallel(moment_model_G, device_ids=[args.gpu])

    for param_q, param_k in zip(model_D.parameters(), moment_model_D.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient
    for param_q, param_k in zip(model_G.parameters(), moment_model_G.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    optimizer_D = AdamW(model_D.parameters(), lr = args.learning_rate, betas = (0.5, 0.999), eps = 1e-8, weight_decay = args.weight_decay)
    optimizer_G = Adam(model_G.parameters(), lr = args.learning_rate, betas = (0.5, 0.999), eps = 1e-8)
    args.start_epoch = 0
    if args.resume:
        args.resume = '{}/last.pth.tar'.format(save_folder_path)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model_D.load_state_dict(checkpoint['state_dict_D'])
            model_G.load_state_dict(checkpoint['state_dict_G'])
            moment_model_D.load_state_dict(checkpoint['moment_state_dict_D'])
            #moment_model_G.load_state_dict(checkpoint['moment_state_dict_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
    ])

    if args.data_type == 'cifar10':
        train_dataset  = torchvision.datasets.CIFAR10(args.data_path, train = True, transform = transform_train, download = True)
        test_dataset  = torchvision.datasets.CIFAR10(args.data_path, train = False, transform = transform_test)
    elif args.data_type == 'cifar100':
        train_dataset  = torchvision.datasets.CIFAR100(args.data_path, train = True, transform = transform_train, download = True)
        test_dataset  = torchvision.datasets.CIFAR100(args.data_path, train = False, transform = transform_test)
    print('NUM of training images: {}'.format(len(train_dataset)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True, drop_last = True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle = False, drop_last = False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last = True, persistent_workers = False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last = False, persistent_workers = False)

    global_batch_size = (torch.distributed.get_world_size() * args.batch_size)
    train_iters_per_epoch = len(train_dataset) // global_batch_size

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if not os.path.exists(save_folder_path):
            os.system('mkdir -p {}'.format(save_folder_path))

    for epoch in range(args.start_epoch, args.epochs + 6):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch % 5 == 0 and epoch != 0:
            eval_model(test_loader, model_D, args, prefix = '')
            eval_model(test_loader, moment_model_D, args, prefix = 'momentum ')
        train(train_loader, test_loader, model_D, moment_model_D, model_G, moment_model_G, optimizer_D, optimizer_G, epoch, args, ngpus_per_node)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.is_master):
            save_checkpoint({
                'epoch': epoch + 1,
                'moment_state_dict_D': moment_model_D.state_dict(),
                'moment_state_dict_G': moment_model_G.state_dict(),
                'state_dict_D': model_D.state_dict(),
                'state_dict_G': model_G.state_dict(),
                'optimizer_D' : optimizer_D.state_dict(),
                'optimizer_G' : optimizer_G.state_dict(),
            }, False, filename = '{}/last.pth.tar'.format(save_folder_path))
            if epoch % 100 == 0 and epoch != 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'moment_state_dict_D': moment_model_D.state_dict(),
                'moment_state_dict_G': moment_model_G.state_dict(),
                'state_dict_D': model_D.state_dict(),
                'state_dict_G': model_G.state_dict(),
                'optimizer_D' : optimizer_D.state_dict(),
                'optimizer_G' : optimizer_G.state_dict(),
                },False , filename = '{}/{:04d}.pth.tar'.format(save_folder_path,epoch))


def train(train_loader, val_loader, model_D, moment_model_D, model_G, moment_model_G, optimizer_D, optimizer_G, epoch, args, ngpus_per_node, scaler = None):
    loss_name = [
                'D_adv_loss', 'G_adv_loss', 'detcov', 'detcov_gen', 'knn_loss', 'sig_val', 'embed_norm', 'D_grad', 'D_param' ,'G_grad' ,'G_param',
                'cls_loss', 'acc',
                'GPU Mem', 'Time']
    moco_loss_meter = [AverageMeter(name, ':6.3f') for name in loss_name]
    progress = ProgressMeter(
        len(train_loader),
        moco_loss_meter,
        prefix="Epoch: [{}]".format(epoch))

    # switc0h to train mode
    t0 = time.time()
    for i, (img, img_label)  in enumerate(train_loader):
        img = img.cuda(args.gpu, non_blocking = True)
        img_label = img_label.cuda(args.gpu, non_blocking = True)
        batch_size = img.shape[0]

        for train_D in [True, False]:

            img.requires_grad = True
            norm_embed, embed, logit, _ = model_D(img)

            z = torch.randn(batch_size, 128, device= args.gpu)
            gen_img = model_G(z)

            if train_D:
                gen_img = gen_img.detach()
                gen_img.requires_grad = True
                gen_norm_embed, gen_embed, _, _ = model_D(gen_img)
                jac_sig = compute_jac_sig(model_D, img, gen_img, embed, gen_embed, iter_num = 1)
                jac_loss = (jac_sig - 1).pow(2).mean()
                embed_norm_hinge_loss = ((embed.norm(dim = -1) - 1) * (embed.norm(dim = -1) < 1).float()).pow(2).mean()
            else:
                gen_norm_embed, gen_embed, _, _ = model_D(gen_img)
            adv_loss, detcov, detcov_gen = compute_adv_loss(norm_embed, gen_norm_embed)

            embed_queue = model_D.module.queue.detach().clone()
            bg_queue = model_D.module.bg_queue.detach().clone()
            with torch.no_grad():
                bg_feat = F.normalize(moment_model_D.module.encoder(torch.cat([img, gen_img], dim = 0)), dim = -1)
                bg_feat, gen_bg_feat = bg_feat.chunk(2)
            knn_feat_sim, knn_rec_sim = compute_knn_loss(bg_feat, norm_embed, gen_bg_feat, gen_norm_embed, embed_queue, bg_queue,
                         args.KNN)
            if train_D:
                cls_loss = torch.nn.CrossEntropyLoss()(logit, img_label)
                cls_acc = (logit.argmax(dim = -1) == img_label).float().mean()
                KNN_loss = -1 * knn_feat_sim + knn_rec_sim
                D_adv_loss = adv_loss
                loss = D_adv_loss + 3 * KNN_loss + 5 * jac_loss + 20 * embed_norm_hinge_loss + cls_loss
                optimizer_D.zero_grad()
                loss.backward()
                D_grad_norm, D_param_norm = prob_gradients(model_D)
                optimizer_D.step()
            else:
                optimizer_G.zero_grad()
                KNN_loss = -1 * knn_rec_sim
                G_adv_loss = adv_loss
                loss = -1 * G_adv_loss + 3 * KNN_loss
                loss.backward()
                G_grad_norm, G_param_norm = prob_gradients(model_G)
                optimizer_G.step()

        for param_q, param_k in zip(model_D.parameters(), moment_model_D.parameters()):
            param_k.data = param_k.data * args.momentum_model + param_q.data * (1. - args.momentum_model)
        for param_q, param_k in zip(model_G.parameters(), moment_model_G.parameters()):
            param_k.data = param_k.data * args.momentum_model + param_q.data * (1. - args.momentum_model)

        model_D.module.dequeue_and_enqueue(norm_embed, bg_feat)

        t1 = time.time()
        val_for_disp = [D_adv_loss, G_adv_loss, detcov, detcov_gen, KNN_loss, jac_sig.abs().mean(), embed.norm(dim = -1).mean(),
                        D_grad_norm, D_param_norm, G_grad_norm, G_param_norm,
                        cls_loss, cls_acc, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), t1 - t0]
        for val_id, val in enumerate(val_for_disp):
            if not isinstance(val, float) and not isinstance(val, int):
                val = val.item()
            moco_loss_meter[val_id].update(val)
        progress.display(i)

    with torch.no_grad():
        rec_img = model_G(z)
    if torch.distributed.get_rank() == 0:
        input_img = (255 * (np.transpose(vutils.make_grid(img[:25].cpu(), padding=2, nrow=5, normalize=False), (1, 2, 0)) * 0.5 + 0.5)).data.numpy().astype(np.uint8)
        rec_img = (255 * (np.transpose(vutils.make_grid(rec_img[:25].cpu(), padding=2, nrow=5, normalize=False), (1, 2, 0)) * 0.5 + 0.5)).data.numpy().astype(np.uint8)
        plt.imshow(input_img)
        plt.savefig('{}/{:05d}_input_img.png'.format(args.save_folder_path, epoch))
        plt.close()
        plt.imshow(rec_img)
        plt.savefig('{}/{:05d}_rec_img.png'.format(args.save_folder_path, epoch))
        plt.close()
    progress.display(i)
    torch.cuda.reset_max_memory_allocated()

@torch.no_grad()
def eval_model(test_loader, model, args, prefix = ''):
    model.eval()
    pred_list = []
    label_list = []
    embed_list = []
    feat_list = []
    for img, label in test_loader:
        img = img.to(args.gpu)
        norm_embed, embed, logit, bg_feat = model(img)
        embed_list.append(embed)
        feat_list.append(bg_feat)
        batch_size = img.shape[0]
        pred = logit.argmax(dim = -1)
        pred_list.append(pred.cpu().data.numpy())
        label_list.append(label.cpu().data.numpy())
    pred = np.concatenate(pred_list)
    label = np.concatenate(label_list).astype(np.int32)
    acc = (pred == label).astype(np.float32).mean()
    raw_embed = torch.cat(embed_list)
    bg_feat = F.normalize(torch.cat(feat_list), dim = -1)
    print('probing test_acc:', acc)
    label = torch.from_numpy(label).to(args.gpu)
    print('kmeans embed:')
    eval_kmeans(raw_embed, label, None, None, args.num_of_cat, args.num_of_cat)
    print('kmeans backbone feature:')
    eval_kmeans(bg_feat, label, None, None, args.num_of_cat, args.num_of_cat)
    model.train()

def compute_jac_sig(encoder, img, gen_img, feat, gen_feat, iter_num = 1):
    """
    Compute the leading singular value of the jacobian matrix.
    """
    def vmap_wrapper(x):
        x = x[None,...]
        out = encoder.module(x)[1]
        return out[0]

    def compute_jvp(x, tangent):
        return torch.func.jvp(vmap_wrapper, (x,), (tangent,))[1]

    cat_feat = torch.cat([feat, gen_feat], dim = 0)
    all_img = torch.cat([img, gen_img], dim = 0)
    u = torch.randn_like(cat_feat)
    with torch.no_grad():
        for _ in range(iter_num):
            v0, v1 = torch.autograd.grad(cat_feat, [img, gen_img], u, create_graph = False, retain_graph = True, allow_unused = True)
            v = torch.cat([v0, v1])
            v = F.normalize(v.flatten(1,-1), dim = -1).reshape((-1, ) + img.shape[1:])
            u = torch.func.vmap(compute_jvp)(all_img, v)
            u_norm = F.normalize(u, dim = -1)
    v0, v1 = torch.autograd.grad(cat_feat, [img, gen_img], u_norm, allow_unused=False,
                           create_graph=True, retain_graph=True,
                           is_grads_batched=False)
    final_v = torch.cat([v0, v1])
    sig_val = torch.einsum('nchw,nchw->n', final_v, v)
    return sig_val

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()
