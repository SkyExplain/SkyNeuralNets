from setuptools import setup, find_packages

setup(
    name="SkyNeuralNets",
    version="0.1.0",
    author="Indira Ocampo",
    author_email="indira.ocampo@csic.es",
    description="A Python package for model selection using CNNs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IndiraOcampo/SkyNeuralNets",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "matplotlib", "healpy", "astropy", "scikit-learn" #Dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

extras_require={
    "gpu": ["tensorflow"],  # if you want to use GPU support
}