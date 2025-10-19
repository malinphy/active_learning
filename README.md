

```markdown
# 🧠 Active Learning for Hate Speech Detection

This repository implements an **Active Learning framework** for text classification tasks using **BERT-based models**.  
It is designed around the problem of detecting **hate speech**, **offensive language**, and **neutral tweets** with minimal labeled data through iterative model improvement.

---

## 🚀 Overview

Active learning allows a model to **selectively query the most informative samples** from an unlabeled pool to be labeled and added to the training set.  
This repository demonstrates how an active learning loop can efficiently improve a text classifier’s performance over multiple cycles.

Each cycle includes:
1. **Model training** on a labeled subset  
2. **Evaluation** on a test set and unlabeled pool  
3. **Sampling** the least confident predictions (e.g., via uncertainty, margin, or entropy)  
4. **Augmenting** the labeled dataset with these new samples  
5. **Repeating** the cycle to continuously improve the model  

---

## 🧩 Repository Structure

```
```
active_learning/
│
├── main.py                              # Main entry point for active learning pipeline
│
├── model/
│   └── classifier_model.py              # BERT-based classifier architecture
│
├── sampling_methods/
│   └── samplings.py                     # Sampling strategies (uncertainty, margin, entropy)
│
├── utils/
│   ├── clean_text.py                    # Text cleaning utilities
│   ├── tweet_dataset.py                 # Custom Dataset class for tweets
│   ├── training.py                      # One-epoch training function
│   ├── prediction.py                    # Evaluation function (accuracy, F1, confusion matrix)
│   └── run_training.py                  # Full training loop per active learning cycle
│
├── data/
│   └── hatespeech/labeled_data.csv      # Dataset (tweet text + labels)
│
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation (this file)

```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/malinphy/active_learning.git
cd active_learning
````

### 2. Create a Virtual Environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📘 Usage

### 1. Prepare the Dataset

Your data should be a CSV file with at least the following columns:

```
['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']
```

The repository expects the file to be located at:

```
data/hatespeech/labeled_data.csv
```

Each tweet is labeled as:

* `0`: Hate Speech
* `1`: Offensive Language
* `2`: Neither

---

### 2. Run the Active Learning Loop

```bash
python main.py
```

This will:

* Initialize a BERT classifier (`google-bert/bert-base-uncased`)
* Train for multiple active learning **cycles**
* Apply **least confidence sampling** to choose new data to label
* Save metrics across all cycles to `active_learning_metrics.pkl`

---

## 🧠 Active Learning Logic

The main loop in `main.py` runs for several **cycles**:

1. **Train** the model on the labeled subset (`balanced_df`)
2. **Evaluate** it on both test data and the unlabeled pool
3. **Compute probabilities** for each unlabeled sample
4. **Select least confident samples** using one of the sampling methods:

   * `least_confidence_sampling`
   * `margin_sampling`
   * `entropy_sampling`
5. **Add those samples** to the training set
6. **Retrain** the model from scratch with the new data
7. **Repeat**

At the end of all cycles, a pickle file (`active_learning_metrics.pkl`) stores:

* Train/Test losses per epoch
* Accuracy and F1 scores
* Sampled indices and probabilities

---

## 🔍 Sampling Strategies

Implemented in [`sampling_methods/samplings.py`](sampling_methods/samplings.py):

| Method                        | Description                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Least Confidence Sampling** | Selects samples where the model has the lowest maximum class probability                              |
| **Margin Sampling**           | Uses the difference between the top two predicted probabilities (smaller margin → higher uncertainty) |
| **Entropy Sampling**          | Uses the entropy of class probability distribution as uncertainty measure                             |

These can be easily swapped in `main.py` by changing:

```python
active_learning_function = least_confidence_sampling
```

to any of the others.

---

## 🏋️ Training Details

* **Base Model:** `google-bert/bert-base-uncased` (from Hugging Face)
* **Optimizer:** AdamW
* **Loss:** CrossEntropyLoss
* **Batch Size:** 32
* **Epochs:** 40
* **Learning Rate:** 1e-6
* **Active Learning Cycles:** 5
* **Query Size per Cycle:** 100 samples

Each cycle fully retrains the model with the expanded labeled dataset.

---

## 📊 Metrics & Outputs

Metrics saved after training include:

* Train/Test Loss
* Train/Test Accuracy
* Train/Test F1 (macro)
* Confusion Matrices
* Sample probabilities and indices for active selection

Output file:

```
active_learning_metrics.pkl
```

You can later load it for analysis or visualization:

```python
import pandas as pd
metrics = pd.read_pickle("active_learning_metrics.pkl")
```

---

## 📈 Example Output

During training you’ll see progress bars such as:

```
🌀 ===== Active Learning Cycle 1/5 =====
===== Epoch 1/40 =====
Train Loss: 0.6801 | Train Acc: 0.7450 | Train F1_macro: 0.7305
Test Loss: 0.6523  | Test Acc: 0.7654 | Test F1_macro: 0.7552
✅ Added 100 new samples to training set.
Remaining pool size: 2580
```



