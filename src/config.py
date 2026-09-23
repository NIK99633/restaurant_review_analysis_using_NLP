class Config:
    # Dataset
    DATASET_PATH = "reviews.tsv"  
    TEST_SIZE = 0.20
    RANDOM_STATE = 0

    # Bag-of-Words
    MAX_FEATURES_BOW = 1500

    # Word2Vec
    W2V_DIM = 100

    # Sequence length (used by Word2Vec sequences and DistilBERT)
    MAX_SEQ_LEN = 40

    # DistilBERT
    BERT_MODEL_NAME = "distilbert-base-uncased"

    # Neural network training
    LSTM_EPOCHS = 8
    FFNN_EPOCHS = 15
    BATCH_SIZE = 16

    # Output
    OUTPUT_DIR = "output"
