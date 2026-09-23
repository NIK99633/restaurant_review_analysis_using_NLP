# Restaurant Review Sentiment Analysis

## Overview
Predicts whether a restaurant review is positive or negative using Python and Naive Bayes. Users can also enter their own review to see the prediction in real-time.

## Dataset
- File: `reviews.tsv` (Tab-separated)  
- Columns: 
  - `Review` → Text of the review
  - `Liked` → Label (0 = Negative, 1 = Positive)
- Example row: `"The food was great!"   1`  
- Save the file and update the path in the code:  
  `dataset = pd.read_csv('C:\\path\\to\\reviews.tsv', delimiter='\t', quoting=3)`  
- Sample dataset: [Kaggle Restaurant Reviews](https://www.kaggle.com/datasets/ishaanv/restaurant-reviews)

## How It Works
1. Preprocess text: remove special characters, lowercase, remove stopwords (except 'not'), stem words.  
2. Convert to numbers using CountVectorizer.  
3. Split data: 80% train, 20% test.  
4. Train Gaussian Naive Bayes model.  
5. Evaluate using confusion matrix and accuracy score.  
6. Predict new reviews: enter text to get ✅ Positive or ❌ Negative.

## How to Run
1. Install dependencies:
   `pip install pandas nltk scikit-learn`
2. Download stopwords

   

# Phase two Overview
## Comparing Bag-of-Words, Word2Vec, and DistilBERT

This project performs binary sentiment analysis on restaurant reviews. Given a review such as *“The food was amazing and the service was excellent,”* the system predicts whether the review expresses **positive** or **negative** sentiment.

The project currently focuses on comparing how different text-representation methods affect sentiment-classification performance. Rather than evaluating a single model, it combines three embedding approaches with three classifier architectures, resulting in **nine experimental combinations**.

## Project Goal

The main research question is:

> How does the choice of text embedding — Bag-of-Words, Word2Vec, or DistilBERT — affect restaurant-review sentiment classification when used with Naive Bayes, LSTM, and feed-forward neural-network classifiers?

The project also investigates which embedding-model combinations are appropriate, efficient, and reliable for a relatively small dataset of restaurant reviews.

## Dataset

The dataset contains approximately **1,000 labelled restaurant reviews** stored in TSV format.

| Column | Description |
|---|---|
| `Review` | Restaurant-review text |
| `Liked` | Sentiment label: `1` = positive and `0` = negative |

The dataset is approximately balanced between positive and negative reviews. Therefore, no oversampling, undersampling, or class-weight adjustment is applied.

Example:

```text
Review                                      Liked
The food was great and the service was fast.  1
The meal was cold and the waiter was rude.    0
```

## Preprocessing

All models use a shared preprocessing pipeline where appropriate:

- Removes non-alphabetic characters
- Converts text to lowercase
- Removes common English stopwords
- Keeps the word `not`, because removing it can reverse meaning; for example, `not good` could become `good`
- Applies Porter stemming to reduce related words to a common form, such as `liked`, `liking`, and `likes` becoming `like`

Raw review text is preserved for DistilBERT because transformers use their own subword tokenizer and pretrained language representation.

## Embedding Methods

The project compares three ways of converting reviews into numerical vectors.

| Embedding | Description |
|---|---|
| **Bag-of-Words** | Represents a review using word counts from the 1,500 most frequent terms. It is fast and simple but does not capture word order or semantic similarity. |
| **Word2Vec** | Learns 100-dimensional static word vectors from the training corpus. A review vector is created by averaging the vectors of its words. |
| **DistilBERT** | Uses a pretrained transformer as a frozen feature extractor. Token-level hidden states are mean-pooled to create one contextual document vector per review. |

## Classifier Models

Each embedding is evaluated with three classifiers.

| Model | Description |
|---|---|
| **Naive Bayes** | A lightweight probabilistic baseline for text classification. |
| **LSTM** | A recurrent neural network designed to process sequential input representations. |
| **Feed-Forward Neural Network** | A small dense neural network that classifies document-level feature vectors. |

Together, the three embeddings and three classifiers create the following 3 × 3 comparison grid:

| Model / Embedding | Bag-of-Words | Word2Vec | DistilBERT |
|---|:---:|:---:|:---:|
| Naive Bayes | ✓ | ✓ | ✓ |
| LSTM | ✓ | ✓ | ✓ |
| Feed-Forward NN | ✓ | ✓ | ✓ |

## Current Findings

The project initially produced unrealistically high results because of train-test data leakage. After correcting the train-test split and ensuring that embeddings were fitted only on training data, the evaluation became more reliable.

- **DistilBERT** is currently the strongest embedding approach, especially when combined with a feed-forward neural network.
- **Bag-of-Words** remains a useful, inexpensive baseline when fast execution and low hardware requirements are important.
- **Word2Vec trained from scratch** performs less well because approximately 1,000 short reviews do not provide enough diverse text to learn high-quality static word embeddings.

A key lesson from this phase is that unusually perfect evaluation scores should be treated as a potential pipeline error rather than immediately accepted as a successful result.

## Outputs

For each dataset experiment, the project generates:

- `results_grid.csv` — Accuracy values for all embedding-model combinations
- `results_heatmap.png` — Visual comparison of the experimental grid
- `results_report.txt` — Summary of results and the best-performing combination
- Confusion matrices for detailed error analysis

## Next Steps

- Fine-tune DistilBERT instead of using it only as a frozen feature extractor
- Compare from-scratch Word2Vec with pretrained Word2Vec or GloVe embeddings
- Add precision, recall, F1-score, and confusion matrices for every combination
- Investigate explainability methods such as LIME or SHAP to identify which review words influence predictions most strongly
