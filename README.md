# 📊 Salary Prediction using Ensemble Machine Learning Models

This project predicts employee salaries using machine learning and ensemble techniques. The model is trained on real-world job data from Kaggle and uses multiple features such as job title, company characteristics, skills, location, and salary ranges.

The final model is a **Voting Regressor** that combines:
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## 📂 Project Structure

```
salary-prediction-ml/
│
├── data/
│   └── salary_data_cleaned.csv
│
├── models/
│   └── salary_prediction_model.pkl
│
├── salaryPrediction.ipynb
│   
│
└── README.md
```

---

## 🎯 Objective

The goal of this project is to build a machine learning model that can predict the average salary for a job using multiple job-related features.

The project includes:
- ✔ Data preprocessing
- ✔ Feature engineering
- ✔ Training multiple ML models
- ✔ Building an ensemble model
- ✔ Model evaluation
- ✔ Saving the trained model
- ✔ Clean and reproducible workflow

---

## 🧪 Dataset

**File Used:** `salary_data_cleaned.csv`

This dataset contains fields such as:
- Job Title
- Company Name
- Location
- Industry, Sector, Type of ownership
- `min_salary`, `max_salary`, `avg_salary`
- `age`, `Rating`
- Skill flags (`python`, `R`, `aws`, `excel`, `spark`)

---

## 🧠 Machine Learning Models Used

### Base Models
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `XGBRegressor`

### Final Model
**Voting Regressor (Ensemble)**

This combines predictions from all three models to improve accuracy and reduce variance.

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Google Colab
- Joblib

---

## 📈 Model Performance

Final metrics obtained from the Voting Regressor:

| Metric | Value | Meaning |
|--------|-------|---------|
| **RMSE** | 1.06 | Average prediction error is ~1 salary unit |
| **MAE** | 0.49 | Model is off by less than 0.5 salary units |
| **R² Score** | 0.9993 | Model explains 99.93% of salary variance |

These results indicate **excellent model performance**.

---

## 🚀 How to Run the Model

### 1. Open the notebook
```
notebook/salaryPrediction.ipynb
```

### 2. Upload the dataset
Place `salary_data_cleaned.csv` inside:
```
data/
```

### 3. Run all cells
Colab will:
- Preprocess the data
- Train multiple models
- Build the ensemble
- Evaluate performance
- Save the final model

---

## 💾 Loading the Saved Model

To use the model in another script:

```python
import joblib

model = joblib.load("models/salary_prediction_model.pkl")
prediction = model.predict(new_data)
print(prediction)
```

---

## 📌 Conclusion

This project demonstrates the full lifecycle of an ML regression problem: from cleaning the data to building an optimized ensemble model. The final model performs exceptionally well and is **production-ready**.
