# Flat Price Prediction – Czech Housing Market

Tento projekt slouží jako případová studie pro predikci cen bytů na českém realitním trhu pomocí machine learningu.

## Cíle projektu

- Načíst a vyčistit data z reálného datasetu
- Prozkoumat a vizualizovat vztahy mezi vlastnostmi bytů a jejich cenou
- Vybudovat prediktivní model pro cenu za m²
- Porovnat různé algoritmy a optimalizovat parametry
- Zajistit čistý, testovatelný a opakovatelný kód
- (Volitelně) vytvořit jednoduchou webovou aplikaci pro predikci

---

## Struktura projektu

.
├── data/ # původní a zpracovaná data
├── notebooks/ # EDA, vizualizace, experimenty
├── src/ # funkce pro načtení, zpracování a modelování
├── tests/ # unit testy
├── scripts/ # spouštěcí nebo pomocné skripty
├── main.py # hlavní skript pro zpracování dat
├── requirements.txt # seznam použitých knihoven
└── README.md

---

## ⚙️ Použité knihovny

- `pandas` – manipulace s daty
- `numpy` – numerické výpočty
- `scikit-learn` – machine learning a preprocessing
- `matplotlib`, `seaborn` – vizualizace
- `jupyter` – analýza a experimenty

---

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

### 3. Spuštění hlavního skriptu

```bash
python main.py
```

## Testy

```bash
python -m unittest discover tests
```
