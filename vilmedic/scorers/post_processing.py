import os
import logging
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import matplotlib.cm as cm
import math


def plot_attention(attentions, pp_dir, seed, logger, split, epoch, dl, smooth=True):
    # Attention are of size num_models x bs x num_layers x num_heads x seq_len x seq_len

    # Some checks
    dataset = copy.deepcopy(dl.dataset)
    if not len(dataset) == len(attentions):
        logger.critical("Attention: This is weird")
        return
    if not hasattr(dataset, 'transform'):
        logger.error("Attention: Dataset does not have a transform attribute")
        return

    out_dir = os.path.join(pp_dir, "attention_{}_{}_{}".format(seed, split, epoch))
    os.makedirs(out_dir, exist_ok=True)

    # Getting rid of the normalization function if any
    new_transform = [t for t in dataset.transform.transforms if 'Normalize' not in t.__class__.__name__]
    dataset.transform.transforms = new_transform

    for sample in dataset:
        image = sample["image"]
        im = transforms.ToPILImage()(image)
        width, height = im.size
        if not width == height:
            logger.error("Attention: image width and height are different")
            return

        plt.imshow(im)
        weights = attentions[-1, -1, -1, -1, -1, :]
        square = int(math.sqrt(len(weights)))

        if smooth:
            alpha_im = skimage.transform.pyramid_expand(attentions[-1, -1, -1, -1, -1, :].reshape(square, square),
                                                        upscale=int(width / square),
                                                        sigma=20)
        else:
            alpha_im = skimage.transform.resize(attentions[-1, -1, -1, -1, -1, :].reshape(square, square),
                                                [width, height])
        plt.imshow(alpha_im, alpha=0.8)
        plt.set_cmap(cm.Reds)
        # plt.clim(0, 1)
        plt.colorbar()
        plt.axis('off')
        print(out_dir)
        plt.savefig(os.path.join(out_dir, "ok"))
        troll


def post_processing(post_processing, results, ckpt_dir, seed, **kwargs):
    if post_processing is None:
        return

    logger = logging.getLogger(str(seed))
    pp_dir = os.path.join(ckpt_dir, 'post_processing')
    os.makedirs(pp_dir, exist_ok=True)
    for pp in post_processing:

        # Plot attention weights
        if "attentions" in pp:
            if 'attentions' not in results:
                logger.warn("No attention weights found in results, skipping")
                continue
            plot_attention(attentions=results['attentions'], pp_dir=pp_dir, seed=seed, logger=logger, **kwargs)

        else:
            logger.warn("Post-processing: No function implemented for {}".format(pp))
