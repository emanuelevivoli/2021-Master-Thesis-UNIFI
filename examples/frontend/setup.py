from setuptools import setup
from setuptools import find_namespace_packages

common_kwargs = dict(
    version="0.0.1",
    license="MIT",
    # install_requires=required,
    long_description=open("README.md").read(),
    author="Emanuele Vivoli",
    author_email="vivoli.emanuele@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">3.6",
    include_package_data=False,
)

setup(
    name="viz",
    packages=find_namespace_packages(where=".*"),
    **common_kwargs
)
