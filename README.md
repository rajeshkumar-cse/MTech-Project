# AI-Driven Clinical Signal Analysis & Interpretability

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-M.Tech_Project-success)

This repository contains the source code for the M.Tech Project (MTP-2) developed by **Rajesh Kumar (M25AI1093)**. 
It features a comprehensive Deep Learning pipeline designed for analyzing medical physiological signals (utilizing `wfdb`). The project goes beyond basic prediction by incorporating state-of-the-art **Explainable AI (XAI)** techniques and automated **Generative AI clinical report generation**.

---

## 📑 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Author](#-author)

---

## 🌟 Features
- **Data Pipeline:** Automated loading and preprocessing of clinical waveform data via the `wfdb` library.
- **Deep Learning Architectures:** Custom PyTorch models optimized for physiological signal classification.
- **Explainable AI (XAI):** In-depth model interpretability using **SHAP**, **Captum**, and **LIME** to ensure clinical trust and transparency.
- **GenAI Clinical Reports:** Automated generation of human-readable clinical reports based on model predictions and XAI outputs.
- **Interactive Dashboard:** Visual analytics dashboard for dynamically exploring model performance and signal data.
- **Benchmarking & Error Analysis:** Comprehensive scripts for evaluating curves, comparing models, and diagnosing errors.

---

## 📂 Project Structure

The project follows a modular and professional standard layout:

```text
MTP-2/
├── models/                     # Saved PyTorch model weights (.pt / .pth)
├── plots/                      # Generated evaluation curves, visualizations, and XAI plots
├── reports/                    # Output directory for generated GenAI clinical reports
├── src/                        # Core source code modules
│   ├── config.py               # Hyperparameters and global configurations
│   ├── dataset.py              # Data loaders and WFDB parsing logic
│   ├── preprocessing.py        # Signal filtering, normalization, and prep
│   ├── augmentation.py         # Data augmentation techniques
│   ├── models.py               # Deep learning network definitions
│   ├── loss.py                 # Custom loss functions
│   ├── train.py                # Main training loop
│   ├── evaluate.py             # Evaluation metrics and validation loop
│   ├── interpretability.py     # Core logic for SHAP/Captum/LIME
│   ├── dashboard.py            # Code for interactive UI dashboard
│   ├── main.py                 # Main entry point to run the pipeline
│   └── generate_*.py           # Dedicated scripts for generating reports, benchmarks, and XAI plots
├── mtp-env/                    # Python virtual environment (ignored in git)
└── requirements.txt            # Project dependencies
```

---

## 🛠 Installation

To set up the development environment locally, follow these steps:

**1. Clone the repository (or navigate to the project folder):**
```bash
cd MTP-2
```

**2. Activate the virtual environment:**
Ensure your `mtp-env` is active.
```bash
# On Windows
mtp-env\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Training the Model
To start the training pipeline, run the main script. *(Configure hyperparameters in `src/config.py` prior to running).*
```bash
python src/main.py
```
*Alternatively, if training is separated into `train.py`:*
```bash
python src/train.py
```

### 2. Generating Explainability (XAI) Plots
To generate LIME explanations or general visual interpretability plots for your trained model:
```bash
python src/generate_XAI_lime_explanation.py
python src/generate_XAI_visualizations.py
```
*The resulting plots will be saved in the `plots/` directory.*

### 3. Generating Clinical Reports
To utilize the GenAI module for drafting clinical reports based on the model's findings:
```bash
python src/generate_GenAI_clinical_report.py
```
*Outputs are saved in the `reports/` folder.*

### 4. Launching the Dashboard
To start the interactive visualization dashboard (e.g., using Streamlit/Dash):
```bash
# Example if using Streamlit:
streamlit run src/dashboard.py

# Example if using standard python script:
python src/dashboard.py
```

---

## 🔬 Methodology Overview
1. **Signal Acquisition:** Raw signals are retrieved using `wfdb`.
2. **Preprocessing:** Noise reduction, segmentation, and normalization (`src/preprocessing.py`).
3. **Training:** Deep learning feature extraction and classification (`src/models.py`, `src/train.py`).
4. **Validation:** Cross-validation and performance benchmarking against standard datasets (`src/generate_DL_benchmark_comparison.py`).
5. **Interpretation:** Highlighting key waveform segments that contributed to the model's decision (`src/interpretability.py`).

---

## 🎓 Author
**Rajesh Kumar**
- Roll Number: M25AI1093
- Program: M.Tech Project (MTP-2)
- Institution: IIT Jodhpur

---
*Note: This repository is intended for academic research and evaluation purposes. Ensure proper data privacy guidelines are followed when utilizing clinical datasets.*
