# Customer Churn Prediction in the Telecom Industry by Handling Class Imbalance

**Author: SamiUllah568**

## Overview

This project aims to predict customer churn in the telecom industry by handling class imbalance. The dataset used for this project is the Telco Customer Churn dataset from Kaggle. The project involves data preprocessing, exploratory data analysis, feature engineering, and model training with hyperparameter tuning.

## Table of Contents

1. [Import Necessary Libraries](#import-necessary-libraries)
2. [Load Dataset](#load-dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Feature Engineering](#feature-engineering)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Conclusion](#conclusion)

## Import Necessary Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')
```

## Load Dataset

```python
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.shape)
df.head(5).T
```

## Data Preprocessing

### Drop Unnecessary Columns

```python
df.drop("customerID", axis=1, inplace=True)
```

### Handle Missing Values

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
```

## Exploratory Data Analysis

### Distribution of Categorical Features

```python
cat_features = df.select_dtypes(exclude='number')
fig, ax = plt.subplots(6, 3, figsize=(14, 25))
ax = ax.flatten()

for i, feature in enumerate(cat_features.columns):
    sns.countplot(x=cat_features[feature], ax=ax[i], palette='Greens_d')
    ax[i].set_title(f"Distribution of {feature}")
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()
```

### Distribution of Numerical Features

```python
num_features = df.select_dtypes(include='number')
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
ax = ax.flatten()

for i, feature in enumerate(num_features.columns):
    sns.histplot(data=num_features, x=feature, ax=ax[i], kde=True, bins=40)
    ax[i].set_title(f"Distribution of {feature} with Churn Status")

plt.tight_layout()
plt.show()
```

## Feature Engineering

### Label Encoding of Target Variable

```python
le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])
```

### One-Hot Encoding for Categorical Features

```python
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)
cat_features = df.select_dtypes(include="object")
encoded_features = ohe.fit_transform(cat_features)
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(cat_features.columns))
df = pd.concat([df.select_dtypes(exclude="object"), encoded_df], axis=1)
```

### Standardization of Numerical Features

```python
scaler = StandardScaler()
num_features = df.select_dtypes(include="number")
scaled_features = scaler.fit_transform(num_features)
scaled_df = pd.DataFrame(scaled_features, columns=num_features.columns)
df = pd.concat([scaled_df, df.select_dtypes(exclude="number")], axis=1)
```

## Model Training and Evaluation

### Splitting Data into Training and Test Sets

```python
X = df.drop("Churn", axis=1)
y = df["Churn"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Balancing the Dataset Using SMOTE

```python
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
```

### Model Training

```python
models = [
    LogisticRegression(),
    GaussianNB(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()
]

def model_train_evaluation(model):
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print(f"Model: {model}")
    print("Train Classification Report:")
    print(classification_report(y_train, train_pred))
    print("Test Classification Report:")
    print(classification_report(y_test, test_pred))
    cm = confusion_matrix(y_test, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

for model in models:
    model_train_evaluation(model)
```

## Hyperparameter Tuning

### Logistic Regression

```python
params = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10],
    'solver': ['lbfgs', 'liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [100, 500],
    'random_state': [42],
}

lr = LogisticRegression()
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
random_search = RandomizedSearchCV(estimator=lr, param_distributions=params, n_iter=100, scoring='recall', cv=cv, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Recall Score:", random_search.best_score_)
```

### XGBoost

```python
params_xgb = {
    'objective': ['binary:logistic'],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [50, 100, 200],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 3],
    'scale_pos_weight': [1, 10, 25],
    'random_state': [42],
}

xgb = XGBClassifier()
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=params_xgb, n_iter=100, scoring='recall', cv=cv, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Recall Score:", random_search.best_score_)
```

## Conclusion

This project demonstrates the process of predicting customer churn in the telecom industry by handling class imbalance. Various machine learning models were trained and evaluated, and hyperparameter tuning was performed to improve model performance. The results show that handling class imbalance and tuning hyperparameters can significantly improve the accuracy and recall of the models.
