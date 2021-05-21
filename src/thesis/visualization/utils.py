# Imports
import os
from datetime import datetime

# Path and constants
from thesis.utils.constants import OUT_PATH
from matplotlib import pyplot as plt


def generate_name(visual_args):
    name = f'd:{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}_'
    extention = '.png'

    # PAPER FIELDs
    name += f"Fie::{visual_args.fields}_"

    # EMBEDDING network
    name += f"Net::{visual_args.model_name}_"

    # PRE
    name += f"PREa::{visual_args.pre_alg}:"
    if visual_args.pre_alg == 'none':
        name = name

    if visual_args.pre_alg == 'umap':
        name += f'{visual_args.pre_n_neighbors}:{visual_args.pre_n_components}:{visual_args.pre_metric}_'

    elif visual_args.pre_alg == 'pca':
        name += f'{visual_args.pre_n_components}_'

    elif visual_args.pre_alg == 'tsne':
        # name += f'{visual_args.pre_perplexity}_{visual_args.pre_n_components}_'
        name += f'{visual_args.pre_n_components}_'

    # CLUSTER
    name += f"Clu::{visual_args.clustering_alg}:"
    if visual_args.clustering_alg == 'kmeans':
        name += f'{visual_args.n_clusters}_'

    if visual_args.clustering_alg == 'hdbscan':
        name += f'{visual_args.min_cluster_size}:{visual_args.metric}:{visual_args.cluster_selection_method}_'

    # POST
    name += f"POSTa::{visual_args.post_alg}:"
    if visual_args.post_alg == 'umap':
        name += f'{visual_args.post_n_neighbors}:{visual_args.post_n_components}:{visual_args.post_metric}:{visual_args.post_min_dist}_'

    elif visual_args.post_alg == 'pca':
        name += f'{visual_args.post_n_components}_'

    elif visual_args.post_alg == 'tsne':
        # name += f'{visual_args.post_perplexity}_{visual_args.post_n_components}_'
        name += f'{visual_args.post_n_components}_'

    return name + extention


def visualization(visual_args, x, y, labels):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Prepare data
    dataframe = {
        'x': x,
        'y': y,
        'labels': labels,
    }
    result = pd.DataFrame(dataframe)

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=2)
    plt.scatter(clustered.x, clustered.y,
                c=clustered.labels, s=2, cmap='rainbow')
    plt.colorbar()
    if not os.path.exists(os.path.join(OUT_PATH, 'imgs')):
        os.makedirs(os.path.join(OUT_PATH, 'imgs'))

    name = generate_name(visual_args)

    plt.savefig(os.path.join(OUT_PATH, 'imgs', name))

    return name
