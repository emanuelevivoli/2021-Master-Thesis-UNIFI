import umap
from sklearn.decomposition import PCA
try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except:
    print("For reasonable computation time, install Multicore-TSNE!")
    from sklearn.manifold import TSNE
import pandas as pd


def pre_reduction(visual_args, embeddings=None):
    if visual_args.pre_alg == 'none':
        print(f"Choosing none, no embedding")
    if visual_args.pre_alg == 'umap':
        embeddings = umap.UMAP(n_neighbors=visual_args.pre_n_neighbors, n_components=visual_args.pre_n_components,
                               metric=visual_args.pre_metric).fit_transform(embeddings)
    elif visual_args.pre_alg == 'pca':
        embeddings = PCA(
            n_components=visual_args.pre_n_components).fit_transform(embeddings)
    elif visual_args.pre_alg == 'tsne':
        embeddings = TSNE(
            n_components=visual_args.pre_n_components).fit_transform(embeddings)
    else:
        raise ValueError(f"Pre-processing Algorithm not supported")

    return embeddings


def post_reduction(visual_args, embeddings=None):
    if visual_args.post_alg == 'none':
        print(f"Choosing none, no embedding")
    if visual_args.post_alg == 'umap':
        embeddings = umap.UMAP(n_neighbors=visual_args.post_n_neighbors, n_components=visual_args.post_n_components,
                               min_dist=visual_args.post_min_dist, metric=visual_args.post_metric).fit_transform(embeddings)
    elif visual_args.post_alg == 'pca':
        embeddings = PCA(
            n_components=visual_args.post_n_components).fit_transform(embeddings)
    elif visual_args.post_alg == 'tsne':
        embeddings = TSNE(
            n_components=visual_args.post_n_components).fit_transform(embeddings)
    else:
        raise ValueError(f"Post-processing Algorithm not supported")

    df = pd.DataFrame.from_records(embeddings)

    return df[0],  df[1]
