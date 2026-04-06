# 🧠 SkyNeuralNets

**A Python toolkit for CMB map model selection using Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red)](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📖 Overview

**SkyNeuralNets** is a modular Python package for **model selection on CMB temperature and polarisation maps (T, Q, U)** using Neural Networks. It provides a full ML pipeline from preprocessing and standardisation to training and evaluation, to classify CMB maps between standard **ΛCDM** and models with **oscillatory features in the primordial power spectrum**.

This package is designed to work with simulated data from [**SkySimulation**](https://github.com/skyexplain/SkySimulation) and feeds into the interpretability analysis in [**SkyInterpret**](https://github.com/skyexplain/SkyInterpret).

This code was developed as part of the analysis pipeline for [*Explaining Neural Networks on the Sky: Machine Learning Interpretability for CMB Maps*](https://arxiv.org/abs/XXXX.XXXXX).

---

## ✨ Features

- 🗂️ **Preprocessing pipeline** — Data preparation utilities for CMB map arrays, including cleaning and formatting for NN input
- 📐 **Standardisation and Normalisation** — Supports both **z-score normalisation** and **per-map standardisation** to ensure consistent input distributions across the dataset
- 🏗️ **Hybrid PCA–MLP architecture** — Implements a combined dimensionality reduction and classification network, compressing high-dimensional CMB maps before classification
- 🎯 **Model selection** — Binary (and multi-class) classification between ΛCDM and non-standard primordial feature models
- 🔧 **Training utilities** — Full training loop with configurable optimisers, loss functions, and early stopping fine-tuning
- 📊 **Monitoring & evaluation** — Track and evaluate model performance with metrics including AUC, validation loss, accuracy, and more

---

## 📦 Installation

### Requirements

- Python ≥ 3.9
- numpy, scipy, matplotlib
- scikit-learn (for PCA)
- TensorFlow *(specify which one you use)*

All dependencies are listed in `requirements.txt`.

### Install

```bash
git clone https://github.com/skyexplain/SkyNeuralNets.git
cd SkyNeuralNets
pip install -e .
```

---

## 🚀 Quick Start
  
See **[Sandbox](https://github.com/skyexplain/Sandbox)** for tutorials and examples.

---

## 🔗 Related Packages

This repository is part of a three-package ecosystem:

| Package | Description |
|---|---|
| [**SkySimulation**](https://github.com/skyexplain/SkySimulation) | CMB map simulation (T, Q, U) with ΛCDM and oscillatory features |
| [**SkyNeuralNets**](https://github.com/skyexplain/SkyNeuralNets) | NN-based model selection on CMB maps *(this repo)* |
| [**SkyInterpret**](https://github.com/skyexplain/SkyInterpret) | Interpretability analysis of the trained neural networks |

---

## 📄 Citation

If you use **SkyNeuralNets** in your research, please cite:

```bibtex
@article{Ocampo2026,
  author        = {Indira Ocampo and Guadalupe Cañas-Herrera},
  title         = {Explaining Neural Networks on the Sky: Machine Learning Interpretability for CMB Maps},
  journal       = {JCAP},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

---

## 📬 Contact

For questions or issues, please open a [GitHub Issue](https://github.com/skyexplain/SkyNeuralNets/issues) or contact [indira.ocampo@csic.es](mailto:indira.ocampo@csic.es).
