import faiss
import pickle
import pandas as pd
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from thesis.app.utils import vector_search
from thesis.utils.constants import DATA_PATH


@st.cache
def read_data(data: str = "example/frontend_example_data.csv"):
    """Read the data from local."""
    data_path = os.path.join(DATA_PATH, data)
    return pd.read_csv(data_path)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="example/frontend_example_index.pickle"):
    """Load and deserialize the Faiss index."""
    index_path = os.path.join(DATA_PATH, path_to_faiss)
    with open(index_path, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def main():
    # Filters
    # for embedding generation
    # ------------------------
    st.sidebar.markdown("**Embeddings**")
    filter_year = st.sidebar.slider(
        "Publication year", 2010, 2021, (2010, 2021), 1)
    filter_citations = st.sidebar.slider("Citations", 0, 250, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

    # Load data and models
    # ------------------------
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("First attempt custom Search-Engine")

    # User search
    # ------------------------
    user_input = st.text_area(  # default query
        "Search box", "physics, low-resource settings")

    # Filters
    st.sidebar.markdown("**Filters**")
    filter_year = st.sidebar.slider(
        "Publication year", 2010, 2021, (2010, 2021), 1)
    filter_citations = st.sidebar.slider("Citations", 0, 250, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

    # Fetch results
    if user_input:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on year
        frame = data[
            (data.year >= filter_year[0])
            & (data.year <= filter_year[1])
            & (data.citations >= filter_citations)
        ]
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(frame.id):
                f = frame[(frame.id == id_)]
            else:
                continue

            st.write(
                f"""**{f.iloc[0].original_title}**  
                **Citations**: {f.iloc[0].citations}  
                **Publication year**: {f.iloc[0].year}  
                **Abstract**
                {f.iloc[0].abstract}
                """
            )


if __name__ == "__main__":
    main()
