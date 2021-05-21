from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="thesis",
    version="0.0.1",
    author="Emanuele Vivoli",
    author_email="vivoli.emanuele@gmail.com",
    description="Master Thesis in NLP, with Huggingface and sentence-transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/emanuelevivoli/thesis",
    project_urls={
        "Bug Tracker": "https://github.com/emanuelevivoli/thesis/issues",
    },
    # download_url="https://github.com/emanuelevivoli/thesis/archive/v0.0.1.zip",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    # install_requires=[
    #     'transformers',
    #     'tqdm',
    #     'pytorch',
    #     'pickle',
    #     'hashlib',
    #     'numpy',
    #     'scikit-learn',
    #     'scipy',
    #     'nltk',
    #     'sentencepiece'
    # ],
    classifiers=[
        "Development Status :: 0 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords="Master-thesis Transformers Networks BERT sentence embedding PyTorch huggingface sBERT NLP deep learning"
)
