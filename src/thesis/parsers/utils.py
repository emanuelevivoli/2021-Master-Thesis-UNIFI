from thesis.parsers.classes import Args


def split_args(args: Args):

    dataset_args = args.datatrain
    training_args = args.training
    model_args = args.model
    embedding_args = args.embedds
    visual_args = args.visual
    run_args = args.runs
    log_args = args.logs

    return dataset_args, training_args, model_args, embedding_args, visual_args, run_args, log_args


def tag_generation(args_: Args):
    """
    Generation of the Run tags (from arguments).
    """
    # empty tags' list
    tags = []

    dataset_args, training_args, model_args, embedding_args, visual_args, run_args, log_args = split_args(
        args_)

    model_args = args_.model
    visual_args = args_.visual

    # PAPER FIELDs
    tags += visual_args.fields

    # EMBEDDING network
    tags += [model_args.model_name_or_path]

    # PRE
    tags += [f'pre.choice: {visual_args.pre.choice}']
    if visual_args.pre.choice == 'UMAP':
        tags += [f'UMAP.pre.n_neighbors: {visual_args.pre.umap.n_neighbors}',
                 f'UMAP.pre.n_components: {visual_args.pre.umap.n_components}',
                 f'UMAP.pre.metric: {visual_args.pre.umap.metric}']

    elif visual_args.pre.choice == 'PCA':
        tags += [f'PCA.pre.n_components: {visual_args.pre.pca.n_components}']

    elif visual_args.pre.choice == 'TSNE':
        tags += [f'TSNE.pre.n_components: {visual_args.pre.tsne.n_components}']

    # CLUSTER
    tags += [f'clust.choice: {visual_args.clust.choice}']
    if visual_args.clust.choice == 'KMEANS':
        tags += [f'KMEANS.n_clusters: {visual_args.clust.kmeans.n_clusters}']

    elif visual_args.clust.choice == 'HDBSCAN':
        tags += [f'HDBSCAN.min_cluster_size: {visual_args.clust.hdbscan.min_cluster_size}',
                 f'HDBSCAN.metric: {visual_args.clust.hdbscan.metric}',
                 f'HDBSCAN.cluster_selection_method: {visual_args.clust.hdbscan.cluster_selection_method}']

    # POST
    tags += [f'post.choice: {visual_args.post.choice}']
    if visual_args.post.choice == 'UMAP':
        tags += [f'UMAP.post.n_neighbors: {visual_args.post.umap.n_neighbors}',
                 f'UMAP.post.n_components: {visual_args.post.umap.n_components}',
                 f'UMAP.post.min_dist: {visual_args.post.umap.min_dist}',
                 f'UMAP.post.metric: {visual_args.post.umap.metric}']

    elif visual_args.post.choice == 'PCA':
        tags += [f'PCA.post.n_components: {visual_args.post.pca.n_components}']

    elif visual_args.post.choice == 'TSNE':
        tags += [f'TSNE.post.n_components: {visual_args.post.tsne.n_components}']

    return tags
