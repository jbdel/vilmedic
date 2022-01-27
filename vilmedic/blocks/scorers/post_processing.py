import tqdm
import os
import logging
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import matplotlib.cm as cm
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import omegaconf
import seaborn as sns

sns.set_theme(style="darkgrid")
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_attention(results, pp_dir, seed, logger, split, epoch, dl, smooth=True, **kwargs):
    if 'attentions' not in results:
        logger.warn("No attention weights found in results, skipping")
        return
    # Attention are of size num_models x bs x num_layers x num_heads x seq_len x seq_len
    attentions = results["attentions"]

    # Some checks
    dataset = copy.deepcopy(dl.dataset)
    if not len(dataset) == len(attentions):
        logger.critical("Attention: This is weird")
        return
    if not hasattr(dataset, 'transform'):
        logger.error("Attention: Dataset does not have a transform attribute")
        return

    out_dir = os.path.join(pp_dir, "plot_attention_{}_{}_{}".format(seed, split, epoch))
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
        plt.savefig(os.path.join(out_dir, "att"))


def plot_representation(keys, results, pp_dir, seed, logger, split, dl, labels_keep=None, max_samples_per_class=None,
                        **kwargs):
    # Getting labels
    attr = [k for k, v in dl.dataset.__dict__.items() if 'LabelDataset' in str(v)]
    label_dataset = getattr(dl.dataset, attr[0])
    labels = label_dataset.labels
    labels_map = label_dataset.labels_map.idx2label
    multi_label = label_dataset.labels_map.multi_label
    out_dir = os.path.join(pp_dir, "plot_representation_{}_{}".format(seed, split))
    os.makedirs(out_dir, exist_ok=True)

    # computing repr
    for key in keys:
        embeddings, emb_labels = list(), list()
        if key not in results:
            logger.warn("Key {} is not found in results dictionary")
            continue

        for vector, label in zip(results[key], labels):
            if multi_label:
                c = np.where(label == 1.)[0]
                if labels_keep is not None:
                    c = [c_ for c_ in c if labels_map[c_] in labels_keep]
                if len(c) != 1:
                    continue  # cant plot a point that belongs to more than one class (or no class from labels_keep)
                label = c[0]

            else:
                if labels_keep is not None:
                    if not labels_map[label] in labels_keep:
                        continue

            emb_labels.append(labels_map[label])
            embeddings.append(vector.numpy())

        emb_labels = np.array(emb_labels)
        embeddings = np.array(embeddings)

        np.save(os.path.join(out_dir, split
                             + '_'
                             + str(key)
                             + '_'
                             + "embeddings"), embeddings)

        np.save(os.path.join(out_dir, split
                             + '_'
                             + str(key)
                             + '_'
                             + "labels"), emb_labels)

        assert len(embeddings) != 0, logging.error("No embedding kept for visualization")

        # Filtering the number of samples per class
        if max_samples_per_class is not None:
            if not isinstance(max_samples_per_class, int):
                logger.warn("Argument max_samples_per_class is not an integer, found {}. Using all points".format(
                    type(max_samples_per_class)))
            else:
                new_labels = []
                new_embeddings = []
                for g in np.unique(emb_labels):
                    ix = np.where(emb_labels == g)[0]
                    np.random.shuffle(ix)
                    new_labels.append(emb_labels[ix[:250]])
                    new_embeddings.append(embeddings[ix[:250]])

                embeddings = np.concatenate(new_embeddings)
                emb_labels = np.concatenate(new_labels)

        # Compute and save plots
        for visualization in [TSNE(n_components=2, n_jobs=4, verbose=0, n_iter=2000),
                              umap.UMAP(n_neighbors=len(labels_map)),
                              ]:

            visualization_name = type(visualization).__name__
            logger.settings('Computing embeddings using {}'.format(visualization_name))

            embeddings_x = visualization.fit_transform(embeddings)

            # Plotting
            fig = plt.figure()
            for g in np.unique(emb_labels):
                ix = np.where(emb_labels == g)
                plt.scatter(embeddings_x[ix, 0], embeddings_x[ix, 1], s=0.1,
                            cmap='Spectral', label=g)

            plt.legend(markerscale=10, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            filename = os.path.join(out_dir, split
                                    + '_'
                                    + str(key)
                                    + '_'
                                    + visualization_name
                                    + '.png')
            fig.savefig(filename)
            plt.close()
            logger.settings('Saved as {}'.format(filename))
    return


def post_processing(post_processing, results, ckpt_dir, seed, dl, **kwargs):
    if post_processing is None:
        return

    logger = logging.getLogger(str(seed))
    pp_dir = os.path.join(ckpt_dir, 'post_processing')
    os.makedirs(pp_dir, exist_ok=True)
    for pp in post_processing:
        if "plot_attention" in pp:  # Plot attention weights
            plot_attention(results=results, pp_dir=pp_dir, seed=seed, logger=logger, **kwargs)
        if "plot_representation" in pp:
            plot_representation(results=results, pp_dir=pp_dir, seed=seed, logger=logger, dl=dl,
                                **pp["plot_representation"], **kwargs)
        else:
            logger.warn("Post-processing: No function implemented for '{}'".format(pp))
