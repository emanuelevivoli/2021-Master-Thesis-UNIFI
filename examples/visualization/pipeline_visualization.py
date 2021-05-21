import os

from thesis.utils.classes import DotDict
from sentence_transformers import SentenceTransformer
from thesis.utils.functions import fuse_datasets_splits, get_dict_args

from thesis.datasets.utils import getting_dataset_splitted

from thesis.utils.constants import (
    ARGS_PATH,
    ARGS_FILE,
    DICTIONARY_FIELD_NAMES,
    _factory_MODELS
)

from thesis.visualization.dim_reduction import pre_reduction, post_reduction
from thesis.visualization.clusterization import clusterization
from thesis.visualization.utils import visualization


def main():
    args = get_dict_args(os.path.join(ARGS_PATH, 'arguments.yaml'))

    datasets = getting_dataset_splitted(args)
    s2orc_chunk = fuse_datasets_splits(datasets)

    args_ = DotDict(args)

    def embedd(args):
        model = SentenceTransformer(args.model_name)
        embeddings = model.encode(
            s2orc_chunk[args.fields], show_progress_bar=True)
        return embeddings
    embeddings = embedd(args_)
    print(
        f"[ BASE EMB ]: {embeddings.shape}   | {args_.model_name} on {args_.fields}")

    embeddings = pre_reduction(args_, embeddings)
    print(f"[ PRE  EMB ]: {embeddings.shape}   | {args_.pre_alg}")

    labels = clusterization(args_, embeddings)
    print(f"[ labels   ]: {labels.shape}       | {args_.clustering_alg}")

    x, y = post_reduction(args_, embeddings)
    print(f"[ POST EMB ]: {x.shape}, {y.shape} | {args_.post_alg}")

    name = visualization(args_,  x, y, labels)
    print(f"[ pic  EMB ]: {name}")


if __name__ == '__main__':
    main()
