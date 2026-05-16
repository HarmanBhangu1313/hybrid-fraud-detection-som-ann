# Hybrid Credit Card Fraud Detection using SOM and ANN

## Overview

This project implements a **hybrid fraud detection pipeline** that combines **unsupervised anomaly detection** with **supervised probabilistic classification** for financial transaction analysis.

The system first applies a **Self-Organizing Map (SOM)** to identify suspicious behavioral clusters in high-dimensional transaction data and then trains an **Artificial Neural Network (ANN)** to generate probabilistic fraud risk scores.

The architecture is designed to simulate real-world fraud analytics workflows where:

* fraud labels are sparse,
* anomalous behavior evolves dynamically,
* and robust risk scoring is more useful than simple binary classification.

---

# Key Highlights

* Built a **hybrid SOM–ANN anomaly detection architecture** for fraud analytics
* Applied **unsupervised transaction pattern analysis** to discover suspicious clusters
* Generated **probabilistic fraud risk scores** using ANN-based classification
* Evaluated model performance using:

  * Precision
  * Recall
  * F1-score
  * ROC-AUC
* Implemented complete preprocessing and scaling workflows for structured financial datasets
* Visualized anomalous regions using SOM U-Matrix mapping

---

# Core Pipeline

```text
Financial Transaction Data
            ↓
Data Preprocessing & Feature Scaling
            ↓
Self-Organizing Map (SOM)
(Unsupervised Anomaly Discovery)
            ↓
Suspicious Cluster Extraction
            ↓
Artificial Neural Network (ANN)
(Supervised Fraud Scoring)
            ↓
Probabilistic Fraud Risk Prediction
```

---

# Tech Stack

## Languages & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* MiniSom
* TensorFlow / Keras

---

# Project Structure

```text
Hybrid-Fraud-Detection-SOM-ANN/
│
├── Hybrid_Model_ANN_SOM.ipynb
├── README.md
└── dataset/
```

---

# Data Preprocessing

The preprocessing pipeline prepares structured financial transaction data for anomaly-aware modeling.

### Preprocessing Steps

* Missing value handling
* Feature normalization using Min-Max Scaling
* Structured transaction feature engineering
* Preparation of ANN-compatible training inputs

The preprocessing stage is critical because SOMs are highly sensitive to feature scaling and distance-based representations.

---

# Self-Organizing Map (SOM)

## Objective

The SOM stage performs **unsupervised anomaly discovery** by projecting high-dimensional transaction vectors onto a low-dimensional topological grid.

### SOM Responsibilities

* Learn transaction distribution patterns
* Detect sparse and abnormal regions
* Identify suspicious behavioral clusters
* Generate anomaly-aware representations

### Mathematical Intuition

For each transaction vector (x), the Best Matching Unit (BMU) is computed as:

[
\text{BMU} = \arg\min_i ||x - w_i||
]

where:

* (x) = input transaction vector
* (w_i) = SOM neuron weight vector

Large inter-neuron distances in the U-Matrix indicate sparse regions that often correspond to anomalous or fraud-like behavior.

---

# Artificial Neural Network (ANN)

## Objective

The ANN stage performs **supervised probabilistic fraud scoring** on suspicious samples identified through SOM-based anomaly discovery.

### ANN Responsibilities

* Learn fraud likelihood patterns
* Generate probabilistic fraud scores
* Improve classification interpretability
* Support downstream fraud prioritization

### Model Characteristics

* Dense feedforward architecture
* Sigmoid activation for probability output
* Binary Cross-Entropy loss
* Supervised learning on anomaly-enriched samples

### Binary Cross-Entropy Loss

[
L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
]

This enables the network to produce calibrated fraud probabilities instead of hard binary predictions.

---

# Evaluation Metrics

The fraud classification stage was evaluated using multiple metrics suitable for imbalanced financial datasets.

## Metrics Used

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

### Why These Metrics Matter

Traditional accuracy can be misleading in fraud detection because fraudulent transactions are rare.

Therefore, the project emphasizes:

* Recall → ability to detect fraudulent activity
* Precision → reduction of false fraud alerts
* F1-score → balance between precision and recall
* ROC-AUC → overall ranking quality of fraud predictions

---

# Visualization

The notebook includes:

* SOM U-Matrix visualization
* Suspicious node mapping
* Fraud probability analysis
* Prediction comparison workflows

These visualizations help interpret how anomalous transaction clusters emerge across the SOM grid.

---

# Results & Insights

## Key Outcomes

* SOM successfully identified sparse anomalous transaction regions
* ANN generated probabilistic fraud risk scores for suspicious samples
* Hybrid learning improved interpretability compared to standalone approaches
* Evaluation metrics demonstrated meaningful fraud classification capability on imbalanced data

---

# Real-World Relevance

This project reflects real-world financial fraud analytics workflows where:

* anomaly detection is often performed before classification,
* fraud labels are limited,
* and risk scoring systems prioritize suspicious transactions probabilistically.

The architecture demonstrates practical ML system design involving:

* unsupervised learning,
* supervised learning,
* anomaly detection,
* feature engineering,
* and fraud analytics evaluation.

---

# Future Improvements

Potential future enhancements include:

* Hyperparameter optimization for SOM grid structure
* Comparison against Isolation Forest and One-Class SVM
* Real-time fraud scoring pipelines
* Threshold optimization using Precision-Recall curves
* Advanced ensemble anomaly detection methods
* Explainable AI (XAI) integration for fraud interpretability

---

# Learning Outcomes

This project strengthened understanding of:

* Hybrid ML architectures
* Financial anomaly detection systems
* Unsupervised-to-supervised ML pipelines
* Probabilistic fraud scoring
* Model evaluation on imbalanced datasets
* Practical AI system engineering

---

# Disclaimer

This project is intended for educational and research-oriented learning purposes and is not designed as a production-grade fraud detection system.
