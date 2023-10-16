import torch.nn as nn
import torch, pdb
import torch.distributed as dist
import numpy as np
import math, itertools
import torch.nn.functional as F

def compute_cov_mu(feat, prob, scale, compute_det = True):
    # feat: N * C
    # prob: N * K
    N,C = feat.shape
    # 1 * K * C
    mean_feat_prob = F.normalize((feat[:,None] * prob[:,:,None]).mean(dim = 0)[None,...], dim = -1)
    # N * K * C
    diff = (feat[:,None] - mean_feat_prob)
    # 1 * C * C
    eye = torch.eye(C, device = feat.device)[None,...]

    # K * C * C
    cov = torch.matmul((diff * prob[:,:,None]).permute(1,2,0), diff.permute(1,0,2)) / (prob.sum(dim = 0)[:,None,None] + 1e-5)
    if compute_det:
        detcov = torch.logdet(eye + scale * cov)
        return mean_feat_prob[0], detcov, cov
    else:
        return mean_feat_prob[0], cov

def compute_mb_dist(feat1, prob1, feat2, prob2, eps = 0.5, cov_inv_eps = 1e-2):
    C = feat1.shape[1]
    if prob1 is None:
        prob1 = torch.ones_like(feat1)[:,[0]]
    if prob2 is None:
        prob2 = torch.ones_like(feat1)[:,[0]]
    N1, K1 = prob1.shape
    N2, K2 = prob1.shape
    scale = C / eps
    eye = torch.eye(C, device = feat1.device)[None,...]

    # K1 * C,  K1, K1 * C * C
    mu1, detcov1, cov1 = compute_cov_mu(feat1, prob1, scale)
    # K2 * C,  K2, K2 * C * C
    mu2, detcov2, cov2 = compute_cov_mu(feat2, prob2, scale)

    feat = torch.cat([feat1, feat2], dim = 0)
    prob = torch.cat([prob1, prob2], dim = 0)

    _, detcov12, cov12 = compute_cov_mu(feat, prob, scale)

    # K1 * K2 * C * C
    #cov12 = 0.5 * (cov1 + cov2)
    # K1 * K2
    #detcov12 = torch.logdet(eye + scale * cov12)

    logdet_term12 = detcov12 - 0.5 * (detcov1 + detcov2)

    mb_dist12 = 0.5 * logdet_term12
    return mb_dist12, detcov1, detcov2

import torch.distributed.nn

def compute_adv_loss(feat, feat_rec):
    mbdist12, detcov1, detcov2 = compute_mb_dist(feat, None, feat_rec, None)
    adv_loss = mbdist12.sum()
    return -1 * adv_loss, detcov1, detcov2

def compute_knn_loss(bg_feat, feat, bg_feat_rec, feat_rec, feat_queue, bg_queue,
             KNN, ms = False, T = 0.5):
    """
    Compute the feature K-Nearest Neighbour Loss
    """
    if KNN == 0:
        return feat.new_zeros(1), feat.new_zeros(1)
    if feat_queue is not None:
        all_feat = torch.cat([feat, feat_queue], dim = 0)
        all_bg_feat = torch.cat([bg_feat, bg_queue], dim = 0)
    else:
        all_feat = feat
        all_bg_feat = bg_feat

    bg_sim = bg_feat_rec @ all_bg_feat.T
    rec_sim = feat_rec @ all_feat.T
    knn_index = bg_sim.topk(KNN + 1, dim = -1)[1][:,1:]
    knn_sim_rec = torch.gather(rec_sim, index = knn_index, dim = -1)
    if ms:
        prob = (knn_sim_rec / T).softmax(dim = -1)
        rec_sim = (knn_sim_rec * prob.detach()).sum(dim = -1).mean()
    else:
        rec_sim = knn_sim_rec.mean()

    bg_sim = bg_feat @ all_bg_feat.T
    sim = feat @ all_feat.T
    knn_index = bg_sim.topk(KNN + 1, dim = -1)[1][:,1:]
    knn_sim = torch.gather(sim, index = knn_index, dim = -1)
    if ms:
        prob = (knn_sim / T).softmax(dim = -1)
        feat_sim = (knn_sim * prob.detach()).sum(dim = -1).mean()
    else:
        feat_sim = knn_sim.mean()
    return feat_sim, rec_sim

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class CatWithMemoryBank(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x, x_mb):
        ctx.save_for_backward(x, x_mb)
        return torch.cat([x, x_mb])

    @staticmethod
    def backward(ctx, grads):
        x, x_mb, = ctx.saved_tensors
        batch_size = x.shape[0]
        mb_size = x_mb.shape[0]
        ratio = 1.0 * (mb_size + batch_size) / batch_size
        return grads[:x.shape[0]] * ratio, grads[x.shape[0]:]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.raw_val = []

    def reset(self):
        self.raw_val = []

    def update(self, val):
        self.val = val
        self.raw_val.append(val)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        data = np.array(self.raw_val)
        return fmtstr.format(name = self.name, val = self.val, avg = data.mean())


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
