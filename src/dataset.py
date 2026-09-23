import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import TextPreprocessor


class ReviewDataset:
    def __init__(self, path: str, preprocessor: TextPreprocessor):
        self.df = pd.read_csv(path, delimiter="\t", quoting=3)
        assert {"Review", "Liked"}.issubset(self.df.columns), \
            "TSV must contain \'Review\' and \'Liked\' columns"

        self.preprocessor = preprocessor
        self.df["tokens"] = self.df["Review"].apply(self.preprocessor.clean)
        self.df["clean_text"] = self.df["tokens"].apply(" ".join)

    def split(self, test_size: float, random_state: int):
        y = self.df["Liked"].values
        train_idx, test_idx = train_test_split(
            self.df.index, test_size=test_size,
            random_state=random_state, stratify=y
        )
        train_df = self.df.loc[train_idx].reset_index(drop=True)
        test_df = self.df.loc[test_idx].reset_index(drop=True)
        return train_df, test_df
