# Imports
import os
from datetime import datetime

# Path and constants
from thesis.utils.constants import OUT_PATH
from matplotlib import pyplot as plt
import pandas as pd


def generate_name(args_):
    name = f'd:{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}_'
    extention = '.png'

    # PAPER FIELDs
    name += f"Fie::{args_.fields}_"

    # EMBEDDING network
    name += f"Net::{args_.model_name_or_path}_"

    # PRE
    name += f"PREa::{args_.pre_alg}:"
    if args_.pre_alg == 'none':
        name = name

    if args_.pre_alg == 'umap':
        name += f'{args_.pre_n_neighbors}:{args_.pre_n_components}:{args_.pre_metric}_'

    elif args_.pre_alg == 'pca':
        name += f'{args_.pre_n_components}_'

    elif args_.pre_alg == 'tsne':
        # name += f'{args_.pre_perplexity}_{args_.pre_n_components}_'
        name += f'{args_.pre_n_components}_'

    # CLUSTER
    name += f"Clu::{args_.clustering_alg}:"
    if args_.clustering_alg == 'kmeans':
        name += f'{args_.n_clusters}_'

    if args_.clustering_alg == 'hdbscan':
        name += f'{args_.min_cluster_size}:{args_.metric}:{args_.cluster_selection_method}_'

    # POST
    name += f"POSTa::{args_.post_alg}:"
    if args_.post_alg == 'umap':
        name += f'{args_.post_n_neighbors}:{args_.post_n_components}:{args_.post_metric}:{args_.post_min_dist}_'

    elif args_.post_alg == 'pca':
        name += f'{args_.post_n_components}_'

    elif args_.post_alg == 'tsne':
        # name += f'{args_.post_perplexity}_{args_.post_n_components}_'
        name += f'{args_.post_n_components}_'

    return name + extention


def visualization(args_, x, y, labels):

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

    name = generate_name(args_)

    plt.savefig(os.path.join(OUT_PATH, 'imgs', name))

    return name
