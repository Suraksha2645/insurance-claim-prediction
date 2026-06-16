# Insurance Claim Prediction System

## Project Overview

This project predicts whether a customer is likely to make an insurance claim using Machine Learning. Multiple classification models were trained and compared, and the best-performing model was selected through evaluation and hyperparameter tuning.

---

## Problem Statement

Insurance companies need to estimate the likelihood of a customer filing a claim. This project uses customer information such as age, BMI, smoking status, region, and medical charges to predict whether a claim will occur.

---

## Dataset Information

### Features

| Feature | Description |
|----------|------------|
| age | Age of customer |
| sex | Gender of customer |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status |
| region | Customer region |
| charges | Medical charges |
| insuranceclaim | Target variable (0 = No Claim, 1 = Claim) |

---

## Project Workflow

### 1. Data Preprocessing

The following preprocessing steps were performed:

- Column name standardization
- Missing value removal using `dropna()`
- Categorical feature identification
- Data validation
- Feature grouping

### 2. Feature Engineering

Additional features were created:

#### BMI_Age

```python
BMI_Age = bmi * age
```

Represents the combined effect of age and BMI.

#### Family_Size

```python
Family_Size = children + 1
```

Represents the total family size.

---

### 3. Exploratory Data Analysis (EDA)

Performed:

- Dataset overview
- Missing value analysis
- Class distribution analysis
- Correlation heatmap
- Feature distribution visualization

---

### 4. Data Splitting

Dataset split into:

```text
Training Data: 80%
Testing Data: 20%
```

Using:

```python
train_test_split(test_size=0.2, random_state=42)
```

---

## Models Used

### Logistic Regression

A baseline linear classification model.

### Random Forest Classifier

An ensemble model using multiple decision trees.

### Gradient Boosting Classifier

A boosting algorithm that builds trees sequentially to improve performance.

---

## Hyperparameter Tuning

The best-performing model was optimized using Grid Search Cross Validation.

Example parameters:

```python
{
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 4, 5]
}
```

Method Used:

```python
GridSearchCV()
```

---

## Model Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

### Confusion Matrix

Measures:

- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

---

## Sample Results

### Model Comparison

| Model | Accuracy |
|---------|----------|
| Logistic Regression | 86.57% |
| Random Forest | 94.78% |
| Gradient Boosting | 95.90% |

### Tuned Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | 97.76% |
| Precision | 98.14% |
| Recall | 98.14% |
| F1 Score | 98.14% |
| ROC-AUC | 99.51% |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Project Structure

```text
insurance-claim-prediction/
│
├── 01_EDA_Preprocessing.ipynb
├── 2_Model_training.ipynb
├── hyperparameter.ipynb
├── 04_final_evaluation.ipynb
│
├── preprocessing.py
├── analysis.py
├── analysid.py
├── model.py
├── main.py
├── app.py
│
├── insurance2.csv
├── processed_insurance.csv
├── best_model.pkl
│
├── requirements.txt
└── README.md
```

---

## Running the Project Locally

### Clone Repository

```bash
git clone https://github.com/Suraksha2645/insurance-claim-prediction.git
```

### Navigate to Project Folder

```bash
cd insurance-claim-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## Streamlit User Interface

The application allows users to:

- Enter customer details
- Calculate prediction probability
- Predict insurance claim status
- View results instantly

---

## Future Improvements

- XGBoost implementation
- Real-time database integration
- AWS deployment
- Enhanced user interface
- Automated model retraining

---

## Author

**Suraksha Sharma**

Machine Learning Project – Insurance Claim Prediction System
