import re

class TextPreprocessor:
    def __init__(self):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        nltk.download("stopwords", quiet=True)
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words("english"))
        self.stopwords.discard("not")  # negation matters for sentiment
    def clean(self, text: str) -> list:
        """Returns a list of cleaned, stemmed tokens for one review."""
        text = re.sub(r"[^a-zA-Z]", " ", text).lower()
        tokens = text.split()
        tokens = [self.stemmer.stem(t) for t in tokens if t not in self.stopwords]
        return tokens
