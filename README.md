# Flat Price Prediction – Czech Housing Market

Tento projekt slouží jako případová studie pro predikci cen bytů na českém realitním trhu pomocí machine learningu.

## Cíle projektu

- Načíst a vyčistit data z reálného datasetu
- Prozkoumat a vizualizovat vztahy mezi vlastnostmi bytů a jejich cenou
- Vybudovat prediktivní model pro cenu za m/2
- Porovnat různé algoritmy a optimalizovat parametry
- Zajistit čistý, testovatelný a opakovatelný kód

---

## Struktura projektu

ey_case_study/

├── data/ # původní a zpracovaná data

├── notebooks/ # EDA, vizualizace, experimenty

├── src/ # funkce pro načtení, zpracování a modelování

├── tests/ # unit testy

├── scripts/ # spouštěcí nebo pomocné skripty

├── main.py # hlavní skript pro zpracování dat

├── requirements.txt # seznam použitých knihoven

└── README.md

---

## Použité knihovny

- `pandas` – manipulace s daty
- `numpy` – numerické výpočty
- `scikit-learn` – machine learning a preprocessing
- `matplotlib`, `seaborn` – vizualizace
- `jupyter` – analýza a experimenty

---

## Průběh řešení

### 1. Načtení dat

- Úprava názvů souborů z DATA PART I na `data_part_1` a PART II na `data_part_2`
- Přejmenování `.txt` na `.csv`
- Spojeno ze dvou `.csv` částí
- Zahozen sloupec `Index`, odstraněny chyby v číselných údajích (unicode mezery, čárky místo teček apod.)
- Vyčištěná data uložena jako `cleaned_data.csv`

### 2. Exploratorní analýza (EDA)

- Rozložení ceny za m/2 napříč kraji
- Závislost ceny na ploše, podlaží, dispozici
- Vývoj cen v čase
- Lokalita a dispozice mají výrazný dopad na cenu

### 3. Předzpracování

- One-hot encoding kategorií
- Standardizace číselných sloupců
- Výsledkem jsou `X_features.csv` a `y_target.csv`

### 4. Výběr a trénink modelů

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**  
  -> Vyhodnocení pomocí MAE, MSE, R^2

  ### 5. Ladění hyperparametrů

- `RandomizedSearchCV` nad Random Forest a Decision Tree
- Uložení nejlepších modelů:
  - `models/random_forest_optimized.joblib`
  - `models/decision_tree_optimized.joblib`
- Ladit Linear Regression nemělo moc smysl
  - málo parametrů

## Výsledky

| Model             | MAE   | R^2 Score |
| ----------------- | ----- | --------- |
| Linear Regression | ~6767 | 0.4479    |
| Decision Tree     | ~6919 | 0.4355    |
| Random Forest     | ~6910 | 0.4563    |

**Random Forest s laděním** dosáhl nejlepší výkonnosti (R^2 ~ 0.46).

## Spuštění projektu

### 1. Klonování repozitáře

```bash
git clone https://github.com/tvoje-username/flat-price-prediction.git
cd flat-price-prediction
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Spuštění hlavního skriptu pro předzpracování dat a vykreslení modelů

```bash
python3 main.py
```

### 4. Spuštění skriptů pro lazení hyperparametrů

```bash
python3 rf_hyperparameters_tuning.py
python3 dt_hyperparameters_tuning.py
```

## Testy

- jsou psané pro funkce ze souboru `preprocessing.py`

```bash
python3 test_preprocessing.py
```
