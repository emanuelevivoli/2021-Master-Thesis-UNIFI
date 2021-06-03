from .base import Config
from typing import Dict
from thesis.utils.classes import DotDict

import json


class VisualConfig(Config):
    """Configuration class for Visualization pipeline management:
    - verbose `verbose` (bool), flag for logging information logs
    - debug `debug` (bool), flag for logging all debug logs ( ⚠️ tons of logs ⚠️ )
    - time `time` (bool), flag for logging time values of functions (when supported)
    - callbacks `callback` (string), is the callback code name (`unused`)
    """

    from logging import Logger

    logger: Logger = None

    def __init__(self, *args, **kwargs):
        print(json.dumps(
            kwargs,
            sort_keys=True,
            indent=4,
            separators=(',', ': ')
        ))

        # inizialize dataset variables
        model = DotDict(kwargs["model"])
        self.model_name_or_path = model.get("model_name_or_path", None)
        self.model_type = model.get("model_type", None)
        self.config_name = model.get("config_name", None)
        assert self.model_name_or_path != None, "ValueError ⚠️ : model_name_or_path must be valid!"

        # inizialize visualization variables
        visual = DotDict(kwargs["visual"])
        self.fields = list(visual.get("fields", []))
        # PRE- dimentionality reduction
        self.pre = DotDict({})
        self.pre.choice = visual.get("pre", {}).get("choice", None)
        assert self.pre.choice != None, "ValueError ⚠️ : pre.choice must be valid!"

        # if self.pre.choice == "umap":
        #     self.pre.umap.n_neighbors = visual["n_neighbors"]
        #     self.pre.umap.metric = visual["metric"]
        #     self.pre.umap.n_components = visual["n_components"]

        # elif self.pre.choice == "tsne":
        #     self.pre.tsne.n_components = visual["n_components"]
        #     self.pre.tsne.metric = visual["metric"]
        #     self.pre.tsne.perplexity = visual["perplexity"]

        # elif self.pre.choice == "pca":
        #     self.pre.pca.n_components = visual["n_components"]

        # CLUSTERIZATION
        self.clust = DotDict({})
        self.clust.choice = visual.get("clust", {}).get("choice", None)

        # if self.clust.choice == "kmeans":
        #     self.clust.kmeans.n_clusters = visual["n_clusters"]
        #     self.clust.kmeans.n_init = visual["n_init"]
        #     self.clust.kmeans.max_iter = visual["max_iter"]

        # elif self.clust.choice == "hdbscan":
        #     self.clust.hdbscan.min_cluster_size = visual["min_cluster_size"]
        #     self.clust.hdbscan.metric = visual["metric"]
        #     self.clust.hdbscan.cluster_selection_method = visual["cluster_selection_method"]

        # elif self.clust.choice == "hierarchical":
        #     self.clust.hierarchical.affinity = visual["affinity"]
        #     self.clust.hierarchical.linkage = visual["linkage"]

        # POST- dimentionality reduction
        self.post = DotDict({})
        self.post.choice = visual.get("post", {}).get("choice", None)

        # if self.post.choice == "umap":
        #     self.post.umap.n_neighbors = visual["n_neighbors"]
        #     self.post.umap.metric = visual["metric"]
        #     self.post.umap.n_components = visual["n_components"]
        #     self.post.umap.min_dist = visual["min_dist"]

        # elif self.post.choice == "tsne":
        #     self.post.tsne.n_components = visual["n_components"]
        #     self.post.tsne.metric = visual["metric"]
        #     self.post.tsne.perplexity = visual["perplexity"]

        # elif self.post.choice == "pca":
        #     self.post.pca.n_components = visual["n_components"]

    def set_logger(self, logger: Logger):
        self.logger = logger

    def get_fingerprint(self) -> Dict:
        # return disctionay of important value to hash
        return dict(self)
