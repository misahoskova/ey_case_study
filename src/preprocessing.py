import pandas as pd
import re

from sklearn.preprocessing import StandardScaler
from typing import Union, IO

def load_merge_data(part1_path: Union[str, IO], part2_path: Union[str, IO], sep: str = ";") -> pd.DataFrame:
    df1 = pd.read_csv(part1_path, sep = sep)
    df2 = pd.read_csv(part2_path, sep = sep, header =None)

    df2.columns = df1.columns
    df = pd.concat([df1, df2], ignore_index=True)
    
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns = ["Index"], errors = "ignore")

    df["Cena za m/2"] = (
        df["Cena za m/2"]
        .astype(str)
        .apply(lambda x: re.sub(r"[^\d.,]", "", x))
        .apply(lambda x: x.replace(",", "."))
        .astype(float)
    )

    df["Místo/čas"] = df["Místo/čas"].astype(str).str.replace(" ", "", regex = False)
    df = df.dropna()
    df = df[(df["Cena za m/2"] > 0) & (df["Obytná plocha"] > 0)] # type: ignore

    return df

def prepare_features(df: pd.DataFrame, target_column: str = "Cena za m/2"):
    X = df.drop(columns = [target_column])
    y = df[target_column]

    categorical_columns = X.select_dtypes(include = ["object"]).columns.tolist()
    X = pd.get_dummies(X, columns = categorical_columns, drop_first = True)

    numeric_columns = X.select_dtypes(include = ["int64", "float64"]).columns.tolist()
    scaler = StandardScaler()
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    return X, y