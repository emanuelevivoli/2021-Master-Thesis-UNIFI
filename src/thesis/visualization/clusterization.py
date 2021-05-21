from sklearn.cluster import KMeans as KMEANS
from hdbscan import HDBSCAN


def clusterization(visual_args, embeddings=None):
    if visual_args.clustering_alg == 'kmeans':
        clustering_model = KMEANS(
            n_clusters=visual_args.n_clusters).fit(embeddings)
        cluster = clustering_model.labels_

    elif visual_args.clustering_alg == 'hdbscan':
        clustering_model = HDBSCAN(min_cluster_size=visual_args.min_cluster_size, metric=visual_args.metric,
                                   cluster_selection_method=visual_args.cluster_selection_method).fit(embeddings)
        cluster = clustering_model.labels_

    else:
        raise ValueError(f"Cluster Algorithm not supported")

    return cluster
