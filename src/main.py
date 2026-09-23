<<<<<<< HEAD
from config import Config
from preprocessing import TextPreprocessor
from dataset import ReviewDataset
from embeddings import BagOfWordsEmbedding, Word2VecEmbedding, DistilBertEmbedding
from models import NaiveBayesModel, LSTMModel, FeedForwardNNModel
from experiment_runner import ExperimentRunner
from report import save_report

def main():
    preprocessor = TextPreprocessor()
    dataset = ReviewDataset(Config.DATASET_PATH, preprocessor)
    train_df, test_df = dataset.split(Config.TEST_SIZE, Config.RANDOM_STATE)
    y_train, y_test = train_df["Liked"].values, test_df["Liked"].values
    embeddings = [
        BagOfWordsEmbedding(max_features=Config.MAX_FEATURES_BOW),
        Word2VecEmbedding(dim=Config.W2V_DIM),
        DistilBertEmbedding(model_name=Config.BERT_MODEL_NAME),
    ]
    models = [
        NaiveBayesModel(),
        LSTMModel(),
        FeedForwardNNModel(),
    ]
    runner = ExperimentRunner(embeddings, models)
    results_df = runner.run(train_df, test_df, y_train, y_test)
    save_report(results_df, Config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
=======
# Import required libraries
import pandas as pd                    
import re                             
import nltk                            
from nltk.corpus import stopwords       # To remove common words (like 'the', 'is')
from nltk.stem import PorterStemmer     # For stemming words (running -> run)
from sklearn.feature_extraction.text import CountVectorizer  # Convert text to numbers
from sklearn.model_selection import train_test_split         # Split dataset
from sklearn.naive_bayes import GaussianNB                  # Naive Bayes model
from sklearn.metrics import confusion_matrix, accuracy_score # Evaluation metrics

# Load dataset (TSV = tab-separated file)
dataset = pd.read_csv(
    'C:\\Users\\N INDRA KARAN\\OneDrive\\Desktop\\Restaurant_Review_analysis\\src\\reviews.tsv',
    delimiter='\t',
    quoting=3
)

# Download stopwords (only first time needed)
nltk.download('stopwords')

# -------------------------
# Text Preprocessing
# -------------------------
corpus = []  # List to store cleaned reviews

for i in range(0, 1000):
    # Remove non-alphabet characters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # Convert to lowercase
    review = review.lower()
    
    # Split sentence into words
    review = review.split()
    
    # Initialize stemmer
    ps = PorterStemmer()
    
    # Get stopwords and keep 'not' (important for sentiment)
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    # Remove stopwords and apply stemming
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    
    # Join words back into a sentence
    review = ' '.join(review)
    
    # Add cleaned review to corpus
    corpus.append(review)

# -------------------------
# Convert Text to Numerical Features
# -------------------------
cv = CountVectorizer(max_features=1500)  
X = cv.fit_transform(corpus).toarray()    

# Target variable (0 = negative, 1 = positive)
y = dataset.iloc[:, 1].values

# -------------------------
# Split Dataset into Train & Test
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# -------------------------
# Train Naive Bayes Model
# -------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on test data
y_pred = gnb.predict(X_test)

# -------------------------
# Evaluate Model
# -------------------------
cm = confusion_matrix(y_test, y_pred)  
print(cm)

accuracy = accuracy_score(y_test, y_pred)  
print(accuracy)

# -------------------------
# User Input Prediction (Real-time testing)
# -------------------------
while True:
    user_input = input("Enter a review (or type 'exit'): ")

    # Exit condition
    if user_input.lower() == 'exit':
        break

    # Preprocess user input (same steps as training)
    review = re.sub('[^a-zA-Z]', ' ', user_input)
    review = review.lower().split()

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    # Convert text to numerical format
    new_X = cv.transform([review]).toarray()

    # Predict sentiment
    prediction = gnb.predict(new_X)

    # Display result
    if prediction[0] == 1:
        print("✅ Positive Review")
    else:
        print("❌ Negative Review")
>>>>>>> 041c22d72377d1b4fc9619c53fdfd463b3159f9d
