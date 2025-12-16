# Hybrid Credit Card Fraud Detection using SOM and ANN

##  Overview
This project implements a **hybrid machine learning pipeline** for **credit card fraud detection**, combining **unsupervised learning** and **supervised learning**.

The system first uses a **Self-Organizing Map (SOM)** to discover anomalous patterns in unlabeled data and then applies an **Artificial Neural Network (ANN)** to assign **probabilistic fraud scores** to selected high-risk samples.

The key motivation is to handle **label scarcity**, a common real-world challenge in fraud detection systems.

---

## Core Idea
Instead of directly training a supervised model on noisy or limited labels, the pipeline follows:
Unlabeled Data
↓
Unsupervised Anomaly Discovery (SOM)
↓
Candidate Selection (Suspicious Regions)
↓
Supervised Scoring (ANN)
↓
Fraud Probability Output

This mirrors how fraud detection systems are often designed in practice.

---

##  Tech Stack
- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **MiniSom**
- **TensorFlow / Keras**
- **Matplotlib**

---

## Project Structure
hybrid-fraud-detection-som-ann/
├── ann_fraud_classifier.ipynb
├── requirements.txt
└── README.md

---

##  Pipeline Details

### Data Preprocessing
- Feature scaling using Min-Max normalization
- Preparation of structured credit card application data

---

###  Unsupervised Anomaly Detection (SOM)
A **Self-Organizing Map (SOM)** is trained to project high-dimensional data onto a 2D grid while preserving topological structure.

The SOM is used to:
- Visualize data density using a **U-Matrix**
- Identify nodes with unusually large inter-neuron distances
- Select **potentially anomalous (fraud-like) samples**

> The SOM component is implemented using the **MiniSom** library.  
> The focus is on **using SOM outputs effectively**, not reimplementing the SOM update rules.

---

###  Supervised Fraud Scoring (ANN)
Samples flagged by the SOM are passed to an **Artificial Neural Network** for supervised learning.

- The ANN outputs **fraud probabilities** (not hard labels)
- Sigmoid activation is used in the output layer
- Binary Cross-Entropy is used as the loss function

---

##  Model Intuition

### SOM (Unsupervised Stage)
For each input vector \(x\), the Best Matching Unit (BMU) is computed as:

BMU = arg min||x-wi||
           i

Nodes with high average distances in the U-Matrix indicate sparse regions, which often correspond to anomalous behavior.

---

### ANN (Supervised Stage)
The ANN minimizes **Binary Cross-Entropy loss**:

L = -[ylog(y^) + (1-y)log(1-y^)}

This allows the model to output **probabilistic fraud scores**, which are more useful than binary predictions in risk-sensitive applications.

---

##  Results
- SOM successfully highlights regions of abnormal behavior
- ANN assigns interpretable fraud likelihoods
- The hybrid approach improves robustness compared to using either method alone

---

##  Notes
- This is a **hybrid unsupervised–supervised learning project**
- No clean fraud labels are assumed initially
- Designed for **learning system-level ML thinking**
- Not intended as a production-ready system

---

##  Future Improvements
- Evaluate performance using labeled fraud data (Precision@K, Recall)
- Compare against Isolation Forest and One-Class SVM
- Automate anomaly threshold selection
- Extend to real-time fraud scoring pipelines
