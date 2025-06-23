import pandas as pd

file_part1 = "../data/DATA PART I.csv"
file_part2 = "../data/PART II.csv"

df1 = pd.read_csv(file_part1, sep=';')
df2 = pd.read_csv(file_part2, sep=';', header=None)

df2.columns = df1.columns

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("../data/flat_prices_combined.csv", index=False, sep=';')

print(f"Dataset načten: {df.shape[0]} řádků, {df.shape[1]} sloupců")
print(df.head())