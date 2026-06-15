# SFENet: Ultra-deep Seismic Weak Signal Enhancement via Nonlocal Self-Similarity Feature Embedding

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of **SFENet** (Structural-aware Noise Estimation Network). 

SFENet is designed to tackle the critical challenges in deep and ultra-deep seismic exploration, specifically the severe amplitude attenuation and the inherent difficulty in distinguishing weak structural signals from complex, nonstationary background noise.

## 🌟 Key Highlights

* **Structural-Aware Noise Estimation:** Unlike general adaptive denoising mechanisms or conventional spatial-domain filters, SFENet uniquely strips away complex nonstationary interference while strictly preserving the integrity and lateral continuity of deep seismic reflection events.
* **Global Subspace Feature Modeling:** Bypasses localized receptive field constraints by projecting features into a low-rank subspace. It leverages nonlocal self-similarity to aggregate long-range structural redundancies across spatially distant traces.
* **Standardized Evaluation Workflow:** The data processing pipeline is strictly optimized for quantitative analysis. Denoised seismic components are directly exported and saved into standard `.npy` array formats, avoiding complex multi-array conversions and ensuring seamless evaluation.

## 🚀 Getting Started

Ensure you have Python 3.8+ installed. The environment requires PyTorch and basic scientific computing libraries.

```bash
# Clone the repository
git clone https://github.com/heyongtian/SFENet-.git
cd SFENet-