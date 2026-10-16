"""Editable installation of the DA-MLDG research source."""

from setuptools import find_packages, setup

setup(
    name="damldg",
    version="2.0.0",
    description="On efficacy of Meta-Learning for Domain Generalization in Speech Emotion Recognition",
    python_requires=">=3.11,<3.14",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch==2.8.0",
        "torchvision==0.23.0",
        "flwr[simulation]==1.40.0",
        "numpy>=2,<3",
        "scipy>=1.14,<2",
        "soundfile>=0.13,<0.14",
        "librosa==0.11.0",
        "pillow>=11,<13",
        "timm>=1.0.20,<1.1",
        "pyarrow>=17,<24",
        "tomli-w>=1.2,<2",
        "huggingface-hub>=0.34,<1",
    ],
    entry_points={"console_scripts": ["damldg=damldg.experiment:main"]},
)
