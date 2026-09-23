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
