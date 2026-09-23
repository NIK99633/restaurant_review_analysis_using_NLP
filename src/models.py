from config import Config

class BaseModel:
    name = "base"
    input_mode = "document"
    def run(self, Xtr, ytr, Xte, yte):
        raise NotImplementedError

class NaiveBayesModel(BaseModel):
    name = "Naive Bayes"
    input_mode = "document"
    def run(self, Xtr, ytr, Xte, yte):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import accuracy_score
        scaler = MinMaxScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        model = GaussianNB()
        model.fit(Xtr_s, ytr)
        preds = model.predict(Xte_s)
        return accuracy_score(yte, preds)

class LSTMModel(BaseModel):
    name = "LSTM"
    input_mode = "sequence"
    def run(self, Xtr, ytr, Xte, yte, epochs=Config.LSTM_EPOCHS):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from sklearn.metrics import accuracy_score
        seq_len, dim = Xtr.shape[1], Xtr.shape[2]
        model = Sequential([
            Input(shape=(seq_len, dim)),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(Xtr, ytr, epochs=epochs, batch_size=Config.BATCH_SIZE,
                  verbose=0, validation_split=0.1)
        preds = (model.predict(Xte, verbose=0) > 0.5).astype(int).flatten()
        return accuracy_score(yte, preds)

class FeedForwardNNModel(BaseModel):
    name = "Feed-Forward NN"
    input_mode = "document"
    def run(self, Xtr, ytr, Xte, yte, epochs=Config.FFNN_EPOCHS):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Dense, Dropout
        from sklearn.metrics import accuracy_score
        dim = Xtr.shape[1]
        model = Sequential([
            Input(shape=(dim,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(Xtr, ytr, epochs=epochs, batch_size=Config.BATCH_SIZE,
                  verbose=0, validation_split=0.1)
        preds = (model.predict(Xte, verbose=0) > 0.5).astype(int).flatten()
        return accuracy_score(yte, preds)
