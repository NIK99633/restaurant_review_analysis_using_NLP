import numpy as np
from config import Config

class BaseEmbedding:
    name = "base"
    def fit(self, train_df):
        raise NotImplementedError
    def document_vectors(self, df):
        raise NotImplementedError
    def sequence_vectors(self, df, max_len=Config.MAX_SEQ_LEN):
        raise NotImplementedError

class BagOfWordsEmbedding(BaseEmbedding):
    name = "Bag-of-Words"
    # BoW has no token order; treat the document vector as a length-1 "sequence" so it can still be fed to the LSTM.
    def __init__(self, max_features=Config.MAX_FEATURES_BOW):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vectorizer = CountVectorizer(max_features=max_features)
    def fit(self, train_df):
        self.vectorizer.fit(train_df["clean_text"])
        return self
    def document_vectors(self, df):
        return self.vectorizer.transform(df["clean_text"]).toarray()
    def sequence_vectors(self, df, max_len=Config.MAX_SEQ_LEN):
        vecs = self.document_vectors(df)
        return vecs.reshape(vecs.shape[0], 1, vecs.shape[1])


class Word2VecEmbedding(BaseEmbedding):
    name = "Word2Vec"
    def __init__(self, dim=Config.W2V_DIM):
        self.dim = dim
        self.model = None
    def fit(self, train_df):
        from gensim.models import Word2Vec
        self.model = Word2Vec(
            sentences=train_df["tokens"].tolist(),
            vector_size=self.dim, window=5, min_count=1, workers=4, seed=0
        )
        return self
    def _vec(self, token):
        return self.model.wv[token] if token in self.model.wv else np.zeros(self.dim)
    def document_vectors(self, df):
        out = []
        for tokens in df["tokens"]:
            vecs = [self._vec(t) for t in tokens]
            out.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.dim))
        return np.array(out)
    def sequence_vectors(self, df, max_len=Config.MAX_SEQ_LEN):
        out = []
        for tokens in df["tokens"]:
            vecs = [self._vec(t) for t in tokens[:max_len]]
            if len(vecs) < max_len:
                vecs += [np.zeros(self.dim)] * (max_len - len(vecs))
            out.append(vecs)
        return np.array(out)


class DistilBertEmbedding(BaseEmbedding):
    name = "DistilBERT"
    def __init__(self, model_name=Config.BERT_MODEL_NAME):
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertModel
        self.torch = torch
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.encoder.eval()  # inference only, no fine-tuning
        self.dim = 768
    def fit(self, train_df):
        return self
    def _hidden_states(self, texts, max_len=Config.MAX_SEQ_LEN):
        enc = self.tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        with self.torch.no_grad():
            out = self.encoder(**enc).last_hidden_state
        return out.numpy()
    def document_vectors(self, df):
        hidden = self._hidden_states(df["Review"].tolist())
        return hidden.mean(axis=1)  # mean pooling over tokens
    def sequence_vectors(self, df, max_len=Config.MAX_SEQ_LEN):
        return self._hidden_states(df["Review"].tolist(), max_len=max_len)