from setuptools import setup, find_packages

setup(
    name="signal-analysis-lab",
    version="0.3.0",
    author="Ashmit Gupta",
    author_email="ashmitg25@gmail.com",
    description=(
        "A self-contained Python library and CLI for multivariate "
        "time-series signal analysis — cross-correlation, spectral "
        "analysis, decomposition, and pairwise lag estimation."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ash-2005/signal-analysis-lab",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.2",
        "h5py>=3.8.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0"],
    },
    entry_points={
        "console_scripts": [
            "siglab=cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Intended Audience :: Science/Research",
    ],
    keywords=[
        "time-series",
        "signal-processing",
        "cross-correlation",
        "lag-estimation",
        "spectral-analysis",
        "neuroimaging",
        "physiology",
    ],
)
