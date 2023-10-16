import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def reassign_centroids(hassign, centroids, rs=None):
    """ reassign centroids when some of them collapse """
    if rs is None:
        rs = np.random
    k, d = centroids.shape
    nsplit = 1
    hassign = hassign.cpu()
    empty_cents = np.where(hassign == 0)[0]

    if empty_cents.size == 0:
        return 0
    print('reassign')
    fac = centroids.new_ones(d)
    fac[::2] += 1 / 1024.
    fac[1::2] -= 1 / 1024.

    # this is a single pass unless there are more than k/2
    # empty centroids
    while empty_cents.size > 0:
        # choose which centroids to split
        probas = hassign.float() - 1
        probas[probas < 0] = 0
        probas /= probas.sum()
        probas = probas.cpu().data.numpy().astype(np.float32)
        nnz = (probas > 0).sum()

        nreplace = min(nnz, empty_cents.size)
        cjs = rs.choice(k, size=nreplace, p=probas)
        for ci, cj in zip(empty_cents[:nreplace], cjs):
            c = centroids[cj]
            centroids[ci] = c * fac
            centroids[cj] = c / fac
            hassign[ci] = hassign[cj] // 2
            hassign[cj] -= hassign[ci]
            nsplit += 1

        empty_cents = empty_cents[nreplace:]

    return nsplit

def kmeans_pp(feat, n_clusters):
    n_local_trials = 2 + int(np.log(n_clusters))
    n, c = feat.shape
    index = torch.randint(low = 0, high = n, size = (1,), device = feat.device).reshape(-1)
    # K * C
    total_index = [index]
    center = feat[[index]]
    closest_dist_sq = torch.cdist(center, feat)
    current_pot = closest_dist_sq.sum()
    for c in range(n_clusters - 1):
        rand_vals = torch.rand(n_local_trials, device= feat.device) * current_pot
        candidate_ids = torch.searchsorted(torch.cumsum(closest_dist_sq.reshape(-1), 0), rand_vals)
        distance_to_candidates = torch.cdist(feat[candidate_ids], feat)
        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=1)
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        total_index.append(best_candidate.item())
    return feat[torch.LongTensor(total_index)]

class torch_kmeans(object):

    @torch.no_grad()
    def __init__(self, feat, num_of_cat, tol = 1e-6, niter = 300, nredo = 5):
        best_inertial = 1e30
        final_assign = None
        final_center =  None
        for _ in range(nredo):
            center = kmeans_pp(feat, num_of_cat)
            old_center = center
            for i in range(niter):
                cdist = torch.cdist(center, feat)
                D, assign = cdist.min(dim = 0)
                assign = torch.nn.functional.one_hot(assign, num_of_cat).float()
                center = (assign.T @ feat) / (assign.sum(dim = 0)[:,None] + 1e-5)
                center = F.normalize(center, dim = -1)
                hassign = assign.sum(dim = 0)
                nsplit = reassign_centroids(hassign, center)
                if (old_center - center).norm() <= tol:
                    break
                old_center = center
            if best_inertial >= D.mean():
                final_assign = assign
                final_center = center
                best_inertial = D.mean()
        self.labels_ = final_assign.argmax(dim = -1)
        self.cluster_centers_ = final_center

@torch.no_grad()
def eval_kmeans(train_feat, train_label, test_feat, test_label, kmeans_cluster, num_of_cat,  prefix = ''):
    kmeans = torch_kmeans(train_feat, kmeans_cluster)
    kmeans_train_label = F.one_hot(kmeans.labels_.long(), kmeans_cluster).float()
    train_label_one_hot = F.one_hot(train_label.long(), kmeans_cluster).float()
    sim = kmeans_train_label.T @ train_label_one_hot
    if num_of_cat == kmeans_cluster:
        kmeans_mapping = torch.from_numpy(linear_sum_assignment(-1 * sim.cpu().data.numpy())[1]).to(train_feat.device)
    else:
        #sim: K * C
        kmeans_mapping = sim.argmax(dim = -1)
    pred_train_label = kmeans_mapping[kmeans.labels_]
    print('kmeans cat = {}, k = {}, train acc ={}'.format(num_of_cat, kmeans_cluster, (pred_train_label == train_label).float().mean().item()))
    if test_feat is not None:
        pred_test_label = kmeans_mapping[torch.cdist(kmeans.cluster_centers_, test_feat).argmin(axis = 0)]
        print('kmeans cat = {}, k = {}, test acc ={}'.format(num_of_cat, kmeans_cluster, pred_test_label == train_label).astype(np.float32).mean())
