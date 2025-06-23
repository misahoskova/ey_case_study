import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from io import StringIO
from preprocessing import load_merge_data, clean_dataframe, prepare_features

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.csv_1 = StringIO(
            "Lokalita;Obytná plocha;Počet místností;Podlaží;Datum prodeje;Místo/čas;Cena za m/2\n"
            "Praha;45;2;3;2023-01-10;Praha 3;105000\n"
            "Brno;60;3;2;2022-11-20;Brno-střed;89000\n"
        )

        self.csv_2 = StringIO(
            "Ostrava;50;2;1;2022-12-15;Ostrava-Jih;75000\n"
        )

        self.df_raw = pd.DataFrame({
            "Lokalita": ["Praha", "Brno", "Ostrava"],
            "Obytná plocha": [45, 60, 50],
            "Počet místností": [2, 3, 2],
            "Podlaží": [3, 2, 1],
            "Datum prodeje": ["2023-01-10", "2022-11-20", "2022-12-15"],
            "Místo/čas": ["Praha 3", "Brno-střed", "Ostrava-Jih"],
            "Cena za m/2": [105000, 89000, 75000]
        })

    def test_load_merge_data(self):
        df = load_merge_data(self.csv_1, self.csv_2)
        self.assertEqual(df.shape, (3, 7))
        self.assertIn("Cena za m/2", df.columns)

    
    def test_clean_dataframe(self):
        df = self.df_raw.copy()
        df.loc[1, 'Obytná plocha'] = None
        cleaned = clean_dataframe(df)
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
        self.assertLess(cleaned.shape[0], df.shape[0])

    
    def test_prepare_features(self):
        df = self.df_raw.copy()
        X, y = prepare_features(df)
        self.assertEqual(len(X), len(y))
        self.assertNotIn('Cena za m/2', X.columns)
        self.assertEqual(y.name, 'Cena za m/2')

if __name__ == "__main__":
    unittest.main()