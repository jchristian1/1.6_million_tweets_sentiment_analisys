# Sentiment Analysis with DistilBERT: End-to-End Project

## Overview
This project focuses on building a sentiment analysis pipeline using the Sentiment140 dataset and leveraging the power of transformer-based architectures (DistilBERT). The pipeline spans from data preprocessing to model evaluation, providing a comprehensive approach to solving a real-world text classification problem.

---

## Project Structure
The project consists of multiple steps organized into Jupyter Notebooks:

1. **Data Cleaning and Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Modeling with Traditional Machine Learning Models**
4. **Fine-Tuning DistilBERT**
5. **Model Evaluation and Error Analysis**

---

## 1. Data Cleaning and Preprocessing
This notebook focuses on cleaning the raw Sentiment140 dataset to prepare it for modeling.

### Steps:
- **Loading the Dataset**: Read the raw dataset and explore its structure.
- **Text Normalization**:
  - Lowercasing all text.
  - Expanding contractions (e.g., "don't" → "do not").
  - Removing special characters, URLs, mentions, and extra whitespaces.
- **Stopword Removal**: Optional for machine learning models but retained for transformers to preserve context.
- **Lemmatization**: Reducing words to their base forms.
- **Duplicate Removal**: Identified and removed duplicate entries based on the `expanded_text` column.
- **Handling Short Texts**: Removed texts with fewer than 5 characters.
- **Final Dataset**: Saved the cleaned dataset as `multipurpose_preprocessed_dataset.csv`.

---

## 2. Exploratory Data Analysis (EDA)
This notebook focuses on understanding the dataset distribution and key insights.

### Steps:
- **Class Distribution**:
  - Balanced the dataset using undersampling for better model performance.
- **Text Length Analysis**:
  - Explored the distribution of tweet lengths.
- **Word Frequency**:
  - Identified the most common words in positive and negative tweets.
- **Visualization**:
  - Used bar plots and word clouds to visualize the insights.

---

## 3. Modeling with Traditional Machine Learning Models
This notebook serves as a baseline comparison for transformer models.

### Models Used:
1. **Logistic Regression**
2. **Multinomial Naive Bayes**
3. **Random Forest**
4. **XGBoost Classifier**

### Steps:
- **Feature Engineering**:
  - Vectorized the cleaned text using TF-IDF and n-grams.
- **Model Training**:
  - Trained the models on the cleaned dataset.
- **Evaluation**:
  - Measured accuracy, precision, recall, and F1-score.
- **Results**:
  - Logistic Regression outperformed other traditional models with 78% accuracy.

---

## 4. Fine-Tuning DistilBERT
This notebook fine-tunes a pretrained DistilBERT model for sentiment analysis.

### Steps:
- **Data Preparation**:
  - Tokenized the dataset using Hugging Face's `AutoTokenizer`.
  - Converted the tokenized outputs into Hugging Face datasets.
- **Model Setup**:
  - Loaded the `distilbert-base-uncased` model for sequence classification.
- **Training**:
  - Fine-tuned the model using Hugging Face's `Trainer` API with the following parameters:
    - Learning rate: `2e-5`
    - Batch size: `16`
    - Epochs: `3`
    - Weight decay: `0.01`
- **Checkpointing**:
  - Saved the best model at the end of training based on the F1-score.

---

## 5. Model Evaluation and Error Analysis
This notebook evaluates the fine-tuned DistilBERT model and identifies areas for improvement.

### Steps:
- **Evaluation Metrics**:
  - Measured accuracy, precision, recall, and F1-score on the test dataset.
  - Achieved **85% accuracy**, significantly outperforming traditional models.
- **Confusion Matrix**:
  - Visualized the confusion matrix to understand misclassifications.
- **Error Analysis**:
  - Examined misclassified samples to identify common patterns.

---

## Results Summary
### Baseline Models Performance:
| Model                     | Accuracy  | Precision  | Recall     | F1 Score   |
|---------------------------|-----------|------------|------------|------------|
| Logistic Regression       | 77.94%    | 76.81%     | 80.05%     | 78.40%     |
| Multinomial Naive Bayes   | 75.27%    | 75.11%     | 75.59%     | 75.35%     |
| Random Forest             | 71.38%    | 70.95%     | 72.41%     | 71.67%     |
| XGBoost Classifier        | 75.93%    | 73.46%     | 81.18%     | 77.13%     |

### DistilBERT Performance:
| Metric                    | Negative Class | Positive Class | Overall  |
|---------------------------|----------------|----------------|----------|
| Precision                 | 85%            | 86%            | 85%      |
| Recall                    | 86%            | 84%            | 85%      |
| F1-Score                 | 86%            | 85%            | 85%      |

---

## Next Steps
1. **Error Analysis**:
   - Perform detailed analysis on misclassified samples to identify patterns.
2. **Hyperparameter Tuning**:
   - Experiment with different learning rates, batch sizes, and epochs.
3. **Larger Models**:
   - Fine-tune more powerful transformers like BERT or RoBERTa.
4. **Domain-Specific Pretraining**:
   - Pretrain the model on a sentiment-specific corpus.
5. **Ensemble Models**:
   - Combine DistilBERT with traditional models to explore ensemble approaches.

---

## Requirements
- Python 3.8+
- Hugging Face Transformers
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- NumPy

Install dependencies using:
```bash
pip install transformers datasets scikit-learn matplotlib seaborn pandas numpy

# 1.6_million_tweets_sentiment_analisys
