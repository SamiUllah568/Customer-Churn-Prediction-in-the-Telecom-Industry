# Customer Churn Prediction in the Telecom Industry by Handling Class Imbalance

**Author:** SamiUllah568

## Overview
This project aims to predict customer churn in the telecom industry by handling class imbalance. The dataset used for this project is the Telco Customer Churn dataset from Kaggle. The project involves data preprocessing, exploratory data analysis, feature engineering, and model training with hyperparameter tuning.
## Table of Contents
1. [About the Dataset](#about-the-dataset)
2. [Import Necessary Libraries](#import-necessary-libraries)
3. [Load Dataset](#load-dataset)
4. [Data Exploration](#data-exploration)
5. [Data Preprocessing](#data-preprocessing)
6. [Feature Engineering](#feature-engineering)
7. [Model Training and Evaluation](#model-training-and-evaluation)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Conclusion](#conclusion)
10. [Save Model](#save-model)

## About the Dataset
[Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

- **gender:** Gender of the customer (e.g., Male, Female).
- **SeniorCitizen:** Indicates if the customer is a senior citizen (e.g., 1: Yes, 0: No).
- **Partner:** Whether the customer has a partner (Yes/No).
- **Dependents:** Whether the customer has dependents (Yes/No).
- **tenure:** Number of months the customer has stayed with the company.
- **PhoneService:** Whether the customer has a phone service (Yes/No).
- **MultipleLines:** Whether the customer has multiple phone lines (Yes/No/No phone service).
- **Contract:** Type of contract the customer has (e.g., Month-to-month, One year, Two year).
- **PaymentMethod:** Payment method used by the customer (e.g., Electronic check, Mailed check, Bank transfer, Credit card).
- **TotalCharges:** Total amount charged to the customer.
- **Churn:** Whether the customer has churned (Yes/No).

## Import Necessary Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings
import os
warnings.filterwarnings('ignore')
```

## Load Dataset
```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape
```

## Data Exploration
```python
df.head(5).T
df.columns
print(df.info())
print(df.describe())
print("Total Missing Rows --->>>  " , df.isnull().sum().sum())
print("Total duplicated Rows --->>>  " , df.duplicated().sum())
for i in df.columns:
    print("Feature -->> ", i)
    print(f"Number of UNIQUE values in {i} columns -- >> [{df[i].nunique()}]")
    print(f"UNIQUE values in {i} columns -- >> [{df[i].unique()}]")
    print("--"*20)
```

## Data Preprocessing
```python
df = df[["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","Contract","TotalCharges","Churn"]]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
```

## Feature Engineering
```python
X = df.drop("Churn", axis=1)
y = df["Churn"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
train_obj = x_train.select_dtypes(include="object")
test_obj = x_test.select_dtypes(include="object")
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)
train_ohe = ohe.fit_transform(train_obj)
test_ohe = ohe.transform(test_obj)
train_columns = ohe.get_feature_names_out(train_obj.columns)
ohe_train = pd.DataFrame(data=train_ohe, columns=train_columns)
ohe_test = pd.DataFrame(data=test_ohe, columns=train_columns)
train_numfeature = x_train.select_dtypes("number")
test_numfeature = x_test.select_dtypes("number")
scaler = StandardScaler()
train_scale = scaler.fit_transform(train_numfeature)
test_scale = scaler.transform(test_numfeature)
scaler_train = pd.DataFrame(data=train_scale, columns=train_numfeature.columns)
scaler_test = pd.DataFrame(data=test_scale, columns=train_numfeature.columns)
x_train = pd.concat([ohe_train, scaler_train], axis=1)
x_test = pd.concat([ohe_test, scaler_test], axis=1)
```

## Model Training and Evaluation
```python
model_lr = LogisticRegression()
model_gnb = GaussianNB()
model_svm = SVC()
model_dt = DecisionTreeClassifier()
model_rf = RandomForestClassifier()
model_ab = AdaBoostClassifier()
model_gb = GradientBoostingClassifier()
model_xgb = XGBClassifier()

def model_train_evaluation(model):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print("Model Performance on Train Data ")
    print("Accuracy Score on Train Data -- >> ", accuracy_score(train_pred, y_train))
    print("Classification Report on Train Data")
    print(classification_report(train_pred, y_train))
    print("Model Performance on Test Data ")
    print("Accuracy Score on Test Data -- >> ", accuracy_score(test_pred, y_test))
    print("Classification Report on Test Data")
    print(classification_report(test_pred, y_test))
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

model_train_evaluation(model_lr)
model_train_evaluation(model_gnb)
model_train_evaluation(model_svm)
model_train_evaluation(model_dt)
model_train_evaluation(model_rf)
model_train_evaluation(model_ab)
model_train_evaluation(model_gb)
model_train_evaluation(model_xgb)
```

## Hyperparameter Tuning
```python
params = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10],
    'solver': ['lbfgs', 'liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [100, 500],
    'random_state': [42],
}
lr_hp = LogisticRegression()
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
random_search = RandomizedSearchCV(
    estimator=lr_hp,
    param_distributions=params,
    n_iter=100,
    scoring='recall',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)
random_search.fit(x_train, y_train)
print("Best Parameters:", random_search.best_params_)
print("Best Recall Score:", random_search.best_score_)
```

## Conclusion
The XGBClassifier performed effectively, achieving a balance between precision and recall across both classes, while demonstrating resilience against overfitting. Below is a detailed evaluation of the model's performance:

### Training Results
- **ROC AUC Score:** 0.7855
- **Classification Report:**
  - **Class 0 (Non-churn):**
    - Precision: 0.84
    - Recall: 0.71
    - F1-Score: 0.77
    - Support: 4,138
  - **Class 1 (Churn):**
    - Precision: 0.75
    - Recall: 0.87
    - F1-Score: 0.80
    - Support: 4,138

### Testing Results
- **ROC AUC Score:** 0.7676
- **Classification Report:**
  - **Class 0 (Non-churn):**
    - Precision: 0.93
    - Recall: 0.69
    - F1-Score: 0.79
    - Support: 1,036
  - **Class 1 (Churn):**
    - Precision: 0.49
    - Recall: 0.85
    - F1-Score: 0.62
    - Support: 373

## Save Model
```python
import pickle

# Save the model
pickle.dump(xgb_s, open("tele_customer_churn_model.pkl", "wb"))

# Load the model
telco_cus_churn_pred_model = pickle.load(open("tele_customer_churn_model.pkl", "rb"))

# Check if the model is loaded correctly
print("Model loaded successfully:", telco_cus_churn_pred_model)
```