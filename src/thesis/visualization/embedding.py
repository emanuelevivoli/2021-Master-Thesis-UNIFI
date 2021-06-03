from thesis.utils.cache import _caching
from sentence_transformers import SentenceTransformer


def embedd(args, corpus, eventual_max_seq_length=512):

    @_caching(
        **args.datatrain.to_dict(),
        **args.training.to_dict(),
        **args.model.to_dict(),
        **args.runs.to_dict(discard=['run_name']),
        function_name='embedd'
    )
    def _embedd(args, corpus, eventual_max_seq_length=512):
        model = SentenceTransformer(args.model.model_name_or_path)
        model.max_seq_length = eventual_max_seq_length if model.max_seq_length is None else model.max_seq_length

        embeddings = model.encode(corpus, show_progress_bar=True)
        return embeddings

    return _embedd(args, corpus, eventual_max_seq_length)
