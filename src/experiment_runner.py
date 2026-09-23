import pandas as pd

class ExperimentRunner:
    def __init__(self, embeddings, models):
        self.embeddings = embeddings
        self.models = models
        self.results = []
    def run(self, train_df, test_df, y_train, y_test):
        for embedding in self.embeddings:
            print(f"\n=== Fitting embedding: {embedding.name} ===")
            embedding.fit(train_df)
            doc_train = embedding.document_vectors(train_df)
            doc_test = embedding.document_vectors(test_df)
            seq_train = embedding.sequence_vectors(train_df)
            seq_test = embedding.sequence_vectors(test_df)
            for model in self.models:
                if model.input_mode == "document":
                    Xtr, Xte = doc_train, doc_test
                else:
                    Xtr, Xte = seq_train, seq_test
                acc = model.run(Xtr, y_train, Xte, y_test)
                print(f"  {model.name:<18s} + {embedding.name:<12s} -> accuracy = {acc:.3f}")
                self.results.append({
                    "Embedding": embedding.name,
                    "Model": model.name,
                    "Accuracy": acc,
                })
        return pd.DataFrame(self.results)
