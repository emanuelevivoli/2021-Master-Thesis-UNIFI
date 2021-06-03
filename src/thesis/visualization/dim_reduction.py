import umap
from sklearn.decomposition import PCA
try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except:
    print("For reasonable computation time, install Multicore-TSNE!")
    from sklearn.manifold import TSNE
import pandas as pd
from thesis.parsers.classes import VisualArguments
import numpy as np
from thesis.utils.cache import _caching


def pre_reduction(visual_args: VisualArguments, embeddings=None):
    @_caching(
        embeddings,
        **visual_args.to_dict(),
        function_name='pre_reduction'
    )
    def _pre_reduction(visual_args: VisualArguments, embeddings=None):
        if visual_args.pre.choice == 'none':
            print(f"Choosing none, no embedding")
        if visual_args.pre.choice == 'UMAP':
            embeddings = umap.UMAP(n_neighbors=visual_args.pre.umap.n_neighbors, n_components=visual_args.pre.umap.n_components,
                                   metric=visual_args.pre.umap.metric).fit_transform(embeddings)
        elif visual_args.pre.choice == 'PCA':
            embeddings = PCA(
                n_components=visual_args.pre.pca.n_components).fit_transform(embeddings)
        elif visual_args.pre.choice == 'TSNE':
            embeddings = TSNE(
                n_components=visual_args.pre.tsne.n_components).fit_transform(embeddings)
        else:
            raise ValueError(f"Pre-processing Algorithm not supported")

        return embeddings

    return _pre_reduction(visual_args, embeddings)


def post_reduction(visual_args: VisualArguments, embeddings=None):
    @_caching(
        embeddings,
        **visual_args.to_dict(),
        function_name='post_reduction'
    )
    def _post_reduction(visual_args: VisualArguments, embeddings=None):
        if visual_args.post.choice == 'none':
            print(f"Choosing none, no embedding")
        if visual_args.post.choice == 'UMAP':
            embeddings = umap.UMAP(n_neighbors=visual_args.post.umap.n_neighbors, n_components=visual_args.post.umap.n_components,
                                   min_dist=visual_args.post.umap.min_dist, metric=visual_args.post.umap.metric).fit_transform(embeddings)
        elif visual_args.post.choice == 'PCA':
            embeddings = PCA(
                n_components=visual_args.post.pca.n_components).fit_transform(embeddings)
        elif visual_args.post.choice == 'TSNE':
            embeddings = TSNE(
                n_components=visual_args.post.tsne.n_components).fit_transform(embeddings)
        else:
            raise ValueError(f"Pre-processing Algorithm not supported")

        df = pd.DataFrame.from_records(embeddings)

        return np.asarray(df[0]),  np.asarray(df[1])

    return _post_reduction(visual_args, embeddings)
