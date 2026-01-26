# 🚦 Traffic Accident Severity Prediction

## 📌 Project Overview
This project uses Machine Learning (Random Forest) to predict the severity of traffic accidents based on weather, road conditions, and time of day. 

**Goal:** Identify key factors causing fatal accidents to improve road safety infrastructure.

## 📊 Key Results
* **Accuracy:** 94% on Test Data
* **Key Insight:** Location (`Start_Lng`, `Start_Lat`) and Time (`Hour`) are stronger predictors of severity than Weather.
* **Impact:** Fatal accidents (Severity 4) are often misclassified as Slight (Severity 2), suggesting that high-severity crashes often occur under "normal" driving conditions (human error).

## 🛠️ Technologies Used
* **Python**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (Random Forest), SMOTE (for Class Imbalance)
* **Visualization**: Matplotlib, Seaborn
* **Data Source**: US Accidents (2016 - 2023) from Kaggle

## 🚀 How to Run
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Process Data:**
    ```bash
    python src/1_process_data.py
    ```
3.  **Train Model:**
    ```bash
    python src/2_train_model.py
    ```
4.  **Visualize Impact:**
    ```bash
    python src/3_visualize_impact.py
    ```

## 📂 Project Structure
* `data/`: Contains raw and processed CSV files.
* `src/`: Python scripts for processing, training, and analysis.
* `models/`: Saved trained models (.pkl).
* `results/`: Generated charts and graphs.