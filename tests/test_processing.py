import unittest
import pandas as pd

from io import StringIO
from src.preprocessing import load_merge_data, clean_dataframe, prepare_features

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.csv_1 = StringIO(
            """id;feature_1;feature_2
            "1;10;100
            "2;20;200
            3;30;300"""
        )

        self.csv_2 = StringIO(
            """id;target
            1;1000
            2;2000
            3;3000"""
        )

        self.df_raw = pd.DataFrame({
            "feature_1": [10, 20, 30],
            "feature_2": [100, 200, 300],
            "target": [1000, 2000, 3000]
        })

    def test_load_merge_data(self):
        df = load_merge_data(self.csv_1, self.csv_2)
        self.assertEqual(df.shape, (3, 4))
        self.assertIn("target", df.columns)

    
    def test_clean_dataframe(self):
        df = self.df_raw.copy()
        df.loc[1, 'feature2'] = None
        cleaned = clean_dataframe(df)
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
        self.assertLess(cleaned.shape[0], df.shape[0])

    
    def test_prepare_features(self):
        df = self.df_raw.copy()
        X, y = prepare_features(df)
        self.assertEqual(len(X), len(y))
        self.assertNotIn('target', X.columns)
        self.assertEqual(y.name, 'target')

if __name__ == "__main__":
    unittest.main()