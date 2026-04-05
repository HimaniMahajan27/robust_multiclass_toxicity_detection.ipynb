<div align="center">

<!-- HEADER GIF -->


# 🧠 Robust Multi-Class Toxic Language Detection
### *Using RoBERTa & Classical ML — A Comparative Study*

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

<br/>

**Author: Himani Mahajan**

</div>

---

## 📌 Overview

<div align="center">

> Building a **robust, scalable multi-class toxicity detection system** for social media text — comparing classical ML baselines against a fine-tuned transformer, with emphasis on balanced performance across an **imbalanced real-world dataset**.

</div>

<br/>

The system classifies user-generated content into **three categories**:

| Label | Class |
|:---:|:---|
| `0` | 🔴 Hate Speech |
| `1` | 🟠 Offensive Language |
| `2` | 🟢 Neither |

---

## 🎯 Objectives

- ✅ Develop an accurate **multi-class toxicity classifier**
- ✅ Compare **classical ML models** vs **transformer-based** architectures
- ✅ Address **class imbalance** through weighted loss strategies
- ✅ Evaluate using **robust metrics** beyond simple accuracy
- ✅ Perform in-depth **error analysis** via confusion matrices

---

## 📂 Dataset

The dataset consists of labeled tweets with the following fields:

| Column | Description |
|:---|:---|
| `text` | Raw tweet content |
| `label` | Target class (0 / 1 / 2) |

> ⚠️ **Data Leakage Prevention:** Columns `count`, `hate_speech`, `offensive_language`, and `neither` were deliberately removed to ensure fair and realistic model evaluation.

---

## ⚙️ Methodology

### 🔹 1. Data Preprocessing

- Removed null values and duplicate entries
- Applied **light text cleaning** — URL & mention normalization
- Preserved linguistic signals: punctuation, casing, slang *(critical for toxicity detection)*

---

### 🔹 2. Exploratory Data Analysis

Analyzed class distribution, text length distribution, and key features:
- Word count per class
- Punctuation density
- Uppercase character ratio

> 📊 *[Add your EDA graphs here — e.g., `images/class_distribution.png`, `images/text_length.png`]*

---

### 🔹 3. Baseline Models — Classical ML

All classical models were implemented using **TF-IDF vectorization**:

| # | Model |
|:---:|:---|
| 1 | Logistic Regression |
| 2 | Linear SVM |
| 3 | Character-level SVM |
| 4 | Combined Word + Char TF-IDF + SVM |

---

### 🔹 4. Primary Model — RoBERTa Transformer

Fine-tuned `roberta-base` with the following configurations:

- 🏗️ Pretrained `roberta-base` encoder
- ⚖️ **Weighted cross-entropy loss** for class imbalance
- ⚡ **Mixed precision training** (GPU-optimized)
- 🛑 **Early stopping** based on validation Macro F1

---

## 📈 Training Dynamics

<div align="center">

*Training vs. Validation Loss across epochs:*
<div align="center">

### 🔵 Training vs Validation Loss

| Epoch | Train Loss | Validation Loss | Trend |
|:----:|:----------:|:---------------:|:------|
| **1** | `0.62` | `0.42` | 🚀 Initial learning |
| **2** | `0.41` | `0.41` | ⚡ Convergence |
| **3** | `0.32` | `0.40` | ✅ Stable model |

</div>

> The model converges smoothly — training loss drops sharply while validation loss stabilizes, confirming **no significant overfitting**.

</div>

---

## 📊 Results

### Final Validation — Best Saved Model

<div align="center">

| Metric | Score |
|:---|:---:|
| **Loss** | 0.4067 |
| **Accuracy** | 0.8838 |
| **Macro F1** ⭐ | **0.7599** |
| **Weighted F1** | 0.8955 |

</div>

---

### All Models Comparison

<div align="center">

| # | Model | Accuracy | Macro F1 ⭐ | Weighted F1 |
|:---:|:---|:---:|:---:|:---:|
| 🥇 | **RoBERTa-base + Weighted Loss** | 0.8838 | **0.7599** | **0.8955** |
| 🥈 | TF-IDF + Logistic Regression | 0.8963 | 0.7406 | 0.8942 |
| 🥉 | Combined Word + Char TF-IDF + Linear SVM | 0.8985 | 0.7369 | 0.8952 |
| 4 | TF-IDF + Linear SVM | 0.8953 | 0.7343 | 0.8929 |
| 5 | Char TF-IDF + Linear SVM | 0.8878 | 0.7255 | 0.8874 |

> ⭐ **Macro F1** is the primary metric — it captures balanced performance across imbalanced classes.

</div>

---

### Confusion Matrix — RoBERTa


#### Key Observations from Confusion Matrix:

| Class | Correct | Misclassified As |
|:---|:---:|:---|
| Hate Speech (186 correct) | 186 | 87 → offensive_language, 13 → neither |
| Offensive Language (3419 correct) | 3419 | 304 → hate_speech, 115 → neither |
| Neither (776 correct) | 776 | 37 → hate_speech, 20 → offensive_language |

---

## 🔍 Key Findings

<div align="center">

| Finding | Detail |
|:---|:---|
| 🏆 **Best Macro F1** | RoBERTa achieves 0.7599 — highest balanced performance across all classes |
| ⚠️ **Majority class bias** | Classical models skew high accuracy by over-predicting offensive language |
| 🔴 **Hardest class** | Hate speech — severe underrepresentation + semantic overlap with offensive language |
| 🤔 **Core challenge** | Contextual ambiguity, implicit toxicity, sarcasm — beyond surface-level patterns |

</div>

---

## ⚠️ Challenges

- 📉 **Severe class imbalance** — hate speech is significantly underrepresented
- 🔀 **Semantic overlap** between hate speech and offensive language
- 🏷️ **Noisy labels** in crowdsourced annotations
- 🎭 **Implicit toxicity** — sarcasm, coded language, and dog-whistles

---



## 💻 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)

</div>

---



## 📌 Conclusion

<div align="center">

> While classical ML models provide **strong, lightweight baselines**, fine-tuned transformer architectures — particularly **RoBERTa with weighted loss** — achieve significantly better **balanced classification** across imbalanced classes.
>
> This work highlights that in complex NLP tasks like toxicity detection, **Macro F1 is the true north star** — accuracy alone masks crucial failures on minority classes like hate speech.

</div>

---

## 🙌 Acknowledgement

<div align="center">

This work is part of **academic research** exploring robust and interpretable toxicity detection systems for real-world social media applications.

<br/>

*⭐ If you found this project insightful, feel free to explore and build upon it!*

</div>
