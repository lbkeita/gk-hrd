# GK HRD - Graph and Knowledge Grounded Health Rumor Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🩺 Overview

GK HRD is a state-of-the-art health rumor detection system that combines:

- **BERT-based text understanding** for accurate claim analysis
- **UMLS knowledge grounding** for medical concept recognition
- **Graph neural networks** for claim-evidence relationships
- **Social propagation modeling** for rumor spread analysis

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **5-Class Classification** | TRUE, FALSE, MIXTURE, NEI, UNPROVEN |
| **Mixture Detection** | 83% recall on nuanced claims |
| **Medical Knowledge** | 1,310 UMLS concepts grounded |
| **Graph Structures** | 29,187 nodes, 27,157 edges |
| **Web Interface** | USTB-branded interactive UI |
| **REST API** | Production-ready endpoints |
| **Batch Processing** | CSV/JSONL support |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 64.6% |
| **Mixture Recall** | 83% |
| **False Detection** | Up to 95% confidence |
| **Average Confidence** | 69.3% |

## 🏗️ Project Structure
