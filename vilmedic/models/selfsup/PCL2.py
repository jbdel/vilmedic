import torch
import json
from random import sample

import faiss
import copy
from torch import Tensor
import math
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import torch.nn as nn
from vilmedic.models.utils import get_n_params
from pl_bolts.models.self_supervised.simclr.simclr_module import Projection
from vilmedic.blocks.vision import *


def nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
    """
    assume out_1 and out_2 are normalized
    out_1: [batch_size, dim]
    out_2: [batch_size, dim]
    """
    # gather representations in case of distributed training
    # out_1_dist: [batch_size * world_size, dim]
    # out_2_dist: [batch_size * world_size, dim]
    out_1_dist = out_1
    out_2_dist = out_2

    # out: [2 * batch_size, dim]
    # out_dist: [2 * batch_size * world_size, dim]
    out = torch.cat([out_1, out_2], dim=0)
    out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

    # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
    # neg: [2 * batch_size]
    cov = torch.mm(out, out_dist.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
    row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + eps)).mean()

    return loss


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


def run_kmeans(x,
               num_cluster,
               temperature,
               ):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results


def evaluation(models, config, dl, **kwargs):
    # No ensembling for this evaluation
    model = models[0]
    visuals = []

    pbar = tqdm(dl, total=len(dl))
    for i, batch in enumerate(pbar):
        out = model(**batch, eval_mode=True)
        visuals.append(out['visual'].cpu())

    # Give PCL class the current evaluation features
    visuals = torch.cat(visuals)
    PCL2.EVAL_FEATURES = copy.deepcopy(visuals.numpy())

    return {'loss': -1,
            'visual': visuals
            }


class PCL2(nn.Module):
    EVAL_FEATURES = None

    def __init__(self, config, clustering_epoch_start, dl, **kwargs):
        super().__init__()

        self.clustering_epoch_start = clustering_epoch_start

        # Visual Encoder
        self.config = copy.deepcopy(config)

        cnn = {
            "proto": "CNN",
            "backbone": "resnet50",
            "output_layer": "avgpool",
            "dropout_out": 0.0,
            "permute": "batch_first",
            "freeze": False
        }

        self.visual = eval(cnn.pop('proto'))(**cnn)
        self.projection = Projection(input_dim=2048, hidden_dim=2048, output_dim=128)
        # Evaluation
        self.eval_func = evaluation

        #
        self.current_epoch = -1
        self.cluster_result = None

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images, index, eval_mode=False, epoch=None, **kwargs):
        if eval_mode:
            return {"visual": self.projection(self.visual(images.cuda()))}

        # New epoch, do clustering
        if epoch + 1 >= self.clustering_epoch_start and epoch != self.current_epoch:
            assert PCL2.EVAL_FEATURES is not None
            self.cluster_result = run_kmeans(PCL2.EVAL_FEATURES,
                                             self.config.num_cluster,
                                             self.config.temperature,
                                             )  # run kmeans clustering on master node
            self.current_epoch = epoch

        images = torch.split(images.cuda(), [3, 3], dim=1)
        z1 = self.projection(self.visual(images[0]))
        z2 = self.projection(self.visual(images[1]))
        loss = nt_xent_loss(z1, z2, self.config.simclr_temperature)

        proto_labels = None
        proto_logits = None
        if self.cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(self.cluster_result['im2cluster'], self.cluster_result['centroids'],
                        self.cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id, self.config.pcl_r)  # sample r negative prototypes
                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

                # compute prototypical logits
                logits_proto = torch.mm(z1, proto_selected.t())

                # targets for prototype assignment
                labels_proto = torch.linspace(0, z1.size(0) - 1, steps=z1.size(0)).long().cuda()

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()], dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

        accp, acc = None, None
        if proto_logits is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(proto_logits, proto_labels):
                loss_proto += self.criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]

            # average loss across all sets of prototypes
            loss_proto /= len(self.config.num_cluster)
            loss += loss_proto

        return {"loss": loss, "custom_print": "accp: {}".format(acc, accp)}

    def __repr__(self):
        s = "PCL\n"
        s += str(json.dumps(OmegaConf.to_container(self.config), indent=4)) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
