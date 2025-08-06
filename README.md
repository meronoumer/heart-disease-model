# Vascular Heart Disease Classification Using Phonocardiogram Audio

## Table of Contents

- [Project Description](#project-description)
- [Project Details](#project-details)
- [Installation and Usage Instructions](#installation-and-usage-instructions)
- [Code Examples and API Documentation](#code-examples-and-api-documentation)
- [Contribution Guidelines](#contribution-guidelines)
- [Acknowledgments and References](#acknowledgments-and-references)

---

## Project Description

**Can we detect vascular heart disease by listening to your heart?**

This project aims to accurately classify specific types of vascular heart diseases—such as Aortic Stenosis (AS), Mitral Regurgitation (MR), and others—using phonocardiogram (PCG) audio signals and machine learning. Leveraging a supervised learning approach, we combine **logistic regression** for classification and **Hidden Markov Models (HMMs)** for temporal segmentation to analyze PCG recordings. Our dataset is sourced from the BUET Multi-Disease Heart Sound (BMD-HS) collection, which includes rich metadata and labeled audio from patients in Dhaka, Bangladesh.

We also place strong emphasis on fairness and generalizability, acknowledging potential biases in our dataset such as demographic representation, geographic skew, and access to care. Our long-term goal is to build a robust, equitable, and interpretable diagnostic tool for use in low-resource clinical settings.

---

## Project Details

- **Technologies Used:**

  - Python 3.11
  - scikit-learn
  - hmmlearn
  - NumPy, pandas, joblib

- **Key Features:**

  - Multi-label classification of heart conditions
  - MFCC feature extraction and statistical representation
  - Logistic Regression + HMM hybrid model
  - Nested cross-validation and performance metrics (F1, Precision-Recall)
  - Bias analysis and dataset metadata utilization

- **Prerequisites:**

  - Python 3.9 or higher
  - pip or conda for dependency management
  - Basic understanding of machine learning and audio signal processing

---

## Installation and Usage Instructions

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/heart-disease-pcg-classifier.git
cd heart-disease-pcg-classifier

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage Instructions

#### 1. Prepare the Dataset

Download the BMD-HS dataset: [BUET Multi-Disease Heart Sound Dataset](https://github.com/sani002/BMD-HS-Dataset)

Ensure your extracted MFCC feature CSV is saved as `extracted_features_df.csv` in the root directory.

#### 2. Run the HMM Classifier

```bash
python hmm_classifier.py
```

This script will:

- Preprocess the dataset
- Train and evaluate HMMs for each condition
- Output evaluation metrics (Micro/Macro F1)
- Save the trained pipeline as `hmm_multilabel_pipeline.joblib`

---

## Code Examples and API Documentation

Here are a few core components from the project:

### HMM Training Example

```python
pos_hmm, neg_hmm = train_pos_neg_hmms(label="AS", idxs=train_indices, k=5)
smooth_transitions(pos_hmm)
```

### Scoring an Example Sequence

```python
def delta_logl_score(pos_hmm, neg_hmm, X_seq):
    return (pos_hmm.score(X_seq) - neg_hmm.score(X_seq)) / len(X_seq)
```

### Model Saving

```python
from joblib import dump
dump({
    "imputer": imp,
    "scaler": scaler,
    "models": final_models,
    "thresholds": thresholds
}, "hmm_multilabel_pipeline.joblib")
```

There are no REST API endpoints in this project, but the pipeline is fully modular for downstream integration.

---

## Contribution Guidelines

We welcome contributions from the community!

### How to Contribute

1. Fork this repository
2. Create a new branch:

```bash
git checkout -b feature/your-feature-name
```

3. Commit your changes:

```bash
git commit -am 'Add new feature'
```

4. Push to your fork:

```bash
git push origin feature/your-feature-name
```

5. Create a Pull Request and describe your changes.

### Reporting Issues

- Use the GitHub Issues tab to report bugs, request enhancements, or suggest features.
- Please include steps to reproduce any issues and relevant log output.

### Coding Conventions

- Follow PEP8 style guidelines.
- Use descriptive variable names.
- Comment complex logic clearly.

---

## Acknowledgments and References

- **Dataset:** [BUET Multi-Disease Heart Sound (BMD-HS)](https://github.com/sani002/BMD-HS-Dataset)
- **Libraries:**
  - [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/)
  - [scikit-learn](https://scikit-learn.org/)
  - [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/)
- **Inspirations and Academic References:**
  - Phonocardiogram-based diagnosis techniques from clinical studies
  - Equitable ML principles in healthcare

---

Feel free to star ⭐ the repo and share if you find this project helpful or inspiring!

