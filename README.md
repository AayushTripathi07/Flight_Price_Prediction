# ✈️ Flight Price Prediction

An end-to-end Machine Learning project to predict airline ticket prices using historical flight data.  
This project demonstrates strong feature engineering, regression modeling, hyperparameter tuning, and model persistence.

---

## 📌 Project Objective

The objective of this project is to build a regression model that can accurately predict flight ticket prices based on journey details such as airline, route, departure time, duration, and number of stops.

This project follows a structured ML workflow similar to real-world industry pipelines.

---

## 📂 Dataset Overview

The dataset contains flight-related information including:

- Airline  
- Date of Journey  
- Source  
- Destination  
- Route  
- Departure Time  
- Arrival Time  
- Duration  
- Total Stops  
- Additional Info  
- Price (Target Variable)

Training Records: **10,682**  
Test Records: **2,671**

---

## 🔄 Machine Learning Pipeline

### 1️⃣ Data Cleaning & Preprocessing

- Removed null values
- Converted date columns to datetime format
- Extracted:
  - Journey Day
  - Journey Month
  - Departure Hour & Minute
  - Arrival Hour & Minute
- Converted duration text (e.g., “2h 50m”) into:
  - Duration_hours
  - Duration_mins
- Dropped redundant columns (Route, Additional_Info, original date/time columns)

---

### 2️⃣ Handling Categorical Variables

- **Nominal Features → One-Hot Encoding**
  - Airline
  - Source
  - Destination

- **Ordinal Feature → Label Encoding**
  - Total_Stops (non-stop = 0, 1 stop = 1, etc.)

Final dataset shape after encoding: **(10682, 30 features)**

---

### 3️⃣ Feature Selection

Methods used:

- Correlation Heatmap
- ExtraTreesRegressor for feature importance

This helped identify the most influential variables affecting price prediction.

---

## 🤖 Model Development

### Model Used:
**Random Forest Regressor**

Why Random Forest?

- Handles non-linear relationships
- Works well without feature scaling
- Reduces overfitting via ensemble learning
- Strong baseline for tabular data

---

## 📊 Model Performance

### Before Hyperparameter Tuning

- R² Score: ~0.79  
- MAE: ~1172  
- RMSE: ~2085  

### After Hyperparameter Tuning (RandomizedSearchCV)

Best Parameters:

- n_estimators: 700  
- max_depth: 20  
- min_samples_split: 15  
- min_samples_leaf: 1  
- max_features: auto  

Final Performance:

- **R² Score: ~0.81**
- MAE: ~1165
- RMSE: ~2015

The model explains approximately **81% of the variance** in ticket prices.

---

## 📈 Evaluation Metrics Used

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Residual Distribution Plot
- Actual vs Predicted Scatter Plot

---

## 💾 Model Persistence

The trained model is saved using pickle for future reuse:

```python
import pickle
file = open("flight_rf.pkl", "wb")
pickle.dump(reg_rf, file)
```
## To load the model:

```python
model = pickle.load(open("flight_rf.pkl", "rb"))
```

## 🛠️ Technologies Used

-Python
-Pandas
-NumPy
-Matplotlib
-Seaborn
-Scikit-learn
-Random Forest
-ExtraTreesRegressor
-RandomizedSearchCV
-Pickle

## ⚠️ Limitations

No real-time pricing API integration
No deployment layer (Flask/Streamlit)
External market variables (fuel cost, demand, holidays) not included
Dataset limited to historical static records


## 🔮 Future Improvements

Deploy as a web application (Flask/Streamlit)
Compare with XGBoost and LightGBM
Add SHAP for model interpretability
Integrate live airline APIs
Build an interactive dashboard


## 👨‍💻 Author

Aayush Tripathi
B.Tech – Computer Science & Engineering
Bennett University
