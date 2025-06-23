from src.preprocessing import load_merge_data, clean_dataframe, prepare_features
from model_selection import run_model_selection

def main():
    df_raw = load_merge_data("data/data_part_1.csv", "data/data_part_2.csv")
    df_cleaned = clean_dataframe(df_raw)

    print("Data loaded and cleaned successfully:", df_cleaned.shape)
    print(df_cleaned.head())

    X, y = prepare_features(df_cleaned)
    print("Input matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    df_cleaned.to_csv("data/cleaned_data.csv", index=False, sep=";")
    X.to_csv("data/X_features.csv", index=False)
    y.to_csv("data/y_target.csv", index=False)

    results = run_model_selection()

    print("\nModel evaluation results:")
    for r in results:
        print(f"{r['name']}: R²={r['r2']:.4f}, MAE={r['mae']:.2f}, MSE={r['mse']:.2f}")

if __name__ == "__main__":
    main()