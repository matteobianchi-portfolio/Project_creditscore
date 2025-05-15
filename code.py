'''
Problem Statement
You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

Task
Given a person’s credit-related information, build a machine learning model that can classify the credit score.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier



def main():
    train_df, test_df = load_data()
    preprocessor, X, y, enc = train_features(train_df)
    pipeline = model(preprocessor, X, y)
    test_df = test_features(test_df)
    prediction(test_df, pipeline, enc)
    
    
def load_data():
    train_df = pd.read_csv(r"train.csv", dtype=({"Monthly_Balance":"str"}))
    test_df = pd.read_csv(r"test.csv")
    train_df = train_df.drop(["Name", "ID","SSN", "Payment_Behaviour"], axis=1)
    test_df = test_df.drop(["Name", "ID","SSN", "Payment_Behaviour"], axis=1)
    return train_df, test_df


def train_features(train_df):
    train_df["Credit_History_Age"] = train_df["Credit_History_Age"].str.extract(r'(\d+)').astype(float)
    columns_to_num = ["Changed_Credit_Limit", "Age", "Credit_History_Age", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance", "Monthly_Inhand_Salary", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment"]
    for colonna in columns_to_num:
        train_df[colonna] = train_df[colonna].astype(str).str.replace("_", "", regex=False)
        train_df[colonna] = pd.to_numeric(train_df[colonna], errors='coerce')
        
    train_df = train_df.dropna(subset=["Credit_Score"])
    train_df = train_df.loc[(train_df["Age"] > 18) & (train_df["Age"] < 100), :]
    train_df = train_df.loc[train_df["Interest_Rate"]<100,:]
    train_df = train_df.loc[train_df["Num_Bank_Accounts"] >= 0,:]
    train_df = train_df.loc[train_df["Num_of_Loan"] >= 0,:]
    train_df = train_df.loc[train_df["Delay_from_due_date"] >= 0,:]
    train_df = train_df.loc[train_df["Num_of_Delayed_Payment"] >= 0,:]
    train_df = train_df.loc[train_df["Changed_Credit_Limit"] >= 0,:]
    
    y = train_df["Credit_Score"]
    X = train_df.drop("Credit_Score", axis=1)
    
    enc = LabelEncoder()
    y_encoded = enc.fit_transform(y)
    
    categorical_columns=X.select_dtypes(include='object').columns.tolist()
    numeric_columns = X.select_dtypes(include=['int64', 'float']).columns.tolist()
            
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    
    preprocessor = ColumnTransformer(transformers=[("categoric", cat_pipeline, categorical_columns),
                                                  ("num", num_pipeline, numeric_columns)])
    
    return preprocessor, X, y_encoded, enc
 
    
def model(preprocessor, X, y):
    
    pipeline = Pipeline(steps=[
        ("prepoc", preprocessor),
        ("model", LogisticRegression())])
    
    param_grid = [
        {
            'model': [RandomForestClassifier()],
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10]
        },
        {
            'model': [LogisticRegression(max_iter=10000)],
            'model__C': [0.1, 1.0, 10.0]
        },
        {
            'model': [XGBClassifier()],
            'model__n_estimators': [20, 60],
            'model__max_depth': [None, 10]
        }
    ]
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    
    print("Best Model:", type(grid_search.best_estimator_.named_steps['model']).__name__)
    print("Best F1:", round(grid_search.best_score_, 2))

    best_pipeline = grid_search.best_estimator_
    return best_pipeline
    
    
def test_features(test_df):
    test_df["Credit_History_Age"] = test_df["Credit_History_Age"].str.extract(r'(\d+)').astype(float)
    columns_to_num = ["Changed_Credit_Limit", "Age", "Credit_History_Age", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance", "Monthly_Inhand_Salary", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment"]
    for col in columns_to_num:
        test_df[col] = test_df[col].astype(str).str.replace("_", "", regex=False)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        
    test_df = test_df.loc[(test_df["Age"] > 18) & (test_df["Age"] < 100), :]
    test_df = test_df.loc[test_df["Interest_Rate"]<60,:]
    test_df = test_df.loc[test_df["Num_Bank_Accounts"] >= 0,:]
    test_df = test_df.loc[test_df["Num_of_Loan"] >= 0,:]
    test_df = test_df.loc[test_df["Delay_from_due_date"] >= 0,:]
    test_df = test_df.loc[test_df["Num_of_Delayed_Payment"] >= 0,:]
    test_df = test_df.loc[test_df["Changed_Credit_Limit"] >= 0,:]
    
    return test_df
    
def prediction(test_df, pipeline, enc):
    print("Predictions on the test dataset: ")
    y_pred = pipeline.predict(test_df)
    y_pred = enc.inverse_transform(y_pred)
    print(pd.Series(y_pred).value_counts(normalize=True))
    
    
if __name__=="__main__":
    main()
