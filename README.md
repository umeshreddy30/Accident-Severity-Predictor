# Accident Severity Predictor

> **A machine learning system that predicts traffic accident severity using weather, location, and time-of-day data — achieving 94% accuracy on 7 years of US accident records.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-US%20Accidents%202016--2023-lightgrey)](https://kaggle.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen)]()

---

## Project Overview

This project uses a **Random Forest classifier** to predict the severity of traffic accidents (scale of 1–4) based on contextual factors. The key finding challenges the intuition that weather is the dominant factor — **location and time of day are far stronger predictors**.

**Goal:** Identify the real root causes of fatal accidents to inform road safety infrastructure investment.

---

## Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | 94% |
| **Dataset Size** | 7.7 million accidents (2016–2023) |
| **Top Predictor** | Start_Lat / Start_Lng (location) |
| **2nd Top Predictor** | Hour of day (time) |
| **Surprising Finding** | Weather ranks lower than expected |

### Critical Insight

Fatal accidents (Severity 4) are frequently misclassified as slight (Severity 2). Investigation revealed that **high-severity crashes often occur under completely normal driving conditions** — clear weather, dry roads, daylight. This suggests **human error and distraction**, not environment, are the primary cause of the worst accidents.

---

## Methodology

```
Raw Data (7.7M rows)
    │
    ▼
Data Cleaning & Feature Engineering
    │  - Handle missing values
    │  - Extract hour/day/month from timestamps
    │  - Encode categorical weather/road conditions
    ▼
Class Imbalance Handling (SMOTE)
    │  - Severity 4 (fatal) is rare → oversample minority class
    ▼
Random Forest Classifier
    │  - 100 estimators
    │  - Feature importance analysis
    ▼
Results & Visualization
    │  - Confusion matrix
    │  - Feature importance bar chart
    │  - Severity distribution maps
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python, Pandas, NumPy | Data processing |
| Scikit-learn | Random Forest, SMOTE |
| Matplotlib, Seaborn | Visualization |
| Kaggle US Accidents dataset | Data source (2016–2023) |

---

## Getting Started

### Install Dependencies

```bash
git clone https://github.com/umeshreddy30/Accident-Severity-Predictor.git
cd Accident-Severity-Predictor
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: Process the raw data
python src/1_process_data.py

# Step 2: Train the model
python src/2_train_model.py

# Step 3: Generate visualizations
python src/3_visualize_impact.py
```

> **Note:** Download the US Accidents dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) and place the CSV in `data/raw/`.

---

## Project Structure

```
Accident-Severity-Predictor/
├── src/
│   ├── 1_process_data.py      # Cleaning, feature engineering
│   ├── 2_train_model.py       # Train Random Forest + SMOTE
│   └── 3_visualize_impact.py  # Charts and confusion matrix
├── models/
│   └── rf_model.pkl           # Saved trained model
├── results/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── severity_distribution.png
├── debug_model.py             # Quick model testing script
└── requirements.txt
```

---

## Feature Importance (Top 5)

```
Start_Lng      ████████████████████  0.21
Start_Lat      ███████████████████   0.19
Hour           ██████████████        0.14
Distance(mi)   █████████             0.09
Temperature    ███████               0.07
```

---

## Broader Impact

This model's output can directly inform:
- **Road infrastructure investment** — identify geographic hotspots
- **Traffic enforcement scheduling** — high-risk hours vs low-risk hours
- **Public safety campaigns** — shift messaging from weather to distraction

---

## Future Work

- [ ] Real-time severity prediction API (FastAPI)
- [ ] Interactive severity heatmap (Folium/Plotly)
- [ ] Incorporate traffic volume data
- [ ] Compare with XGBoost and LightGBM
