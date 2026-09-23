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
2. Download stopwords: