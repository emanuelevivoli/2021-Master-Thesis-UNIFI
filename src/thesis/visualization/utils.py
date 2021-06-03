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

    model_args = args_.model
    visual_args = args_.visual

    # PAPER FIELDs
    name += f"Fie::{':'.join(visual_args.fields)}_"

    # EMBEDDING network
    name += f"Net::{model_args.model_name_or_path.replace('/', ':').replace('_', '.')}_"

    # PRE
    name += f"PREa::{visual_args.pre.choice}:"
    if visual_args.pre.choice == 'none':
        name = name

    if visual_args.pre.choice == 'UMAP':
        name += f'{visual_args.pre.umap.n_neighbors}:{visual_args.pre.umap.n_components}:{visual_args.pre.umap.metric}_'

    elif visual_args.pre.choice == 'PCA':
        name += f'{visual_args.pre.pca.n_components}_'

    elif visual_args.pre.choice == 'TSNE':
        # name += f'{visual_args.pre..tsne.perplexity}_{visual_args.pre..tsne.metric}_'
        name += f'{visual_args.pre.tsne.n_components}_'

    # CLUSTER
    name += f"Clu::{visual_args.clust.choice}:"
    if visual_args.clust.choice == 'kmeans':
        name += f'{visual_args.clust.kmeans.n_clusters}_'

    if visual_args.clust.choice == 'hdbscan':
        name += f'{visual_args.clust.hdbscan.min_cluster_size}:{visual_args.clust.hdbscan.metric}:{visual_args.clust.hdbscan.cluster_selection_method}_'

    # POST
    name += f"POSTa::{visual_args.post.choice}:"
    if visual_args.post.choice == 'UMAP':
        name += f'{visual_args.post.umap.n_neighbors}:{visual_args.post.umap.n_components}:{visual_args.post.umap.metric}:{visual_args.post.umap.min_dist}_'

    elif visual_args.post.choice == 'PCA':
        name += f'{visual_args.post.pca.n_components}_'

    elif visual_args.post.choice == 'TSNE':
        # name += f'{visual_args.post.tsne.perplexity}_{visual_args.post.tsne.metric}_'
        name += f'{visual_args.post.tsne.n_components}_'

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

    return name, plt
