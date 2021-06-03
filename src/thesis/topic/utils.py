import numpy as np


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = np.round(np.asarray(tf_idf.T), 3)
    indices = tf_idf_transposed.argsort()[:, -n:]
    words_top_n_words = {label: [words[j]
                                 for j in indices[i]][::-1] for i, label in enumerate(labels)}
    tfidf_top_n_words = {label: [tf_idf_transposed[i][j]
                                 for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return words_top_n_words, tfidf_top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes
