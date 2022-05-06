import torch
import json
import faiss
import copy

from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import torch.nn as nn
from vilmedic.models.utils import get_n_params
from vilmedic.blocks.vision.selfsup import MoCo


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
    PCL.EVAL_FEATURES = copy.deepcopy(visuals.numpy())

    return {'loss': -1,
            'visual': visuals
            }


class PCL(nn.Module):
    EVAL_FEATURES = None

    def __init__(self, config, clustering_epoch_start, dl, **kwargs):
        super().__init__()

        self.clustering_epoch_start = clustering_epoch_start

        # Visual Encoder
        self.config = copy.deepcopy(config)
        self.model = MoCo(base_encoder=config.arch,
                          dim=config.low_dim,
                          r=config.pcl_r,
                          m=config.moco_m,
                          T=config.temperature,
                          mlp=config.mlp)
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        # Evaluation
        self.eval_func = evaluation

        #
        self.current_epoch = -1
        self.cluster_result = None

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images, index, eval_mode=False, epoch=None, **kwargs):
        if eval_mode:
            return {"visual": self.model(im_q=images.cuda(),
                                         is_eval=True)}

        images = torch.split(images.cuda(), [3, 3], dim=1)
        # New epoch, do clustering
        if epoch + 1 >= self.clustering_epoch_start and epoch != self.current_epoch:
            assert PCL.EVAL_FEATURES is not None
            self.cluster_result = run_kmeans(PCL.EVAL_FEATURES,
                                             self.config.num_cluster,
                                             self.config.temperature,
                                             )  # run kmeans clustering on master node
            self.current_epoch = epoch

        output, target, output_proto, target_proto = self.model(im_q=images[0],
                                                                im_k=images[1],
                                                                cluster_result=self.cluster_result,
                                                                index=index)
        accp, acc = None, None
        loss = self.criterion(output, target)
        if output_proto is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(output_proto, target_proto):
                loss_proto += self.criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]

            # average loss across all sets of prototypes
            loss_proto /= len(self.config.num_cluster)
            loss += loss_proto

        acc = accuracy(output, target)[0]
        return {"loss": loss, "custom_print": "acc: {}, accp: {}".format(acc, accp)}

    def __repr__(self):
        s = "PCL\n"
        s += str(json.dumps(OmegaConf.to_container(self.config), indent=4)) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
