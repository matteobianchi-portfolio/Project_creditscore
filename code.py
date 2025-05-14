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



def main():
    train_df, test_df = load_data()
    preprocessor, train_df = train_features(train_df)
    pipeline, enc = model(preprocessor, train_df)
    test_df = test_features(test_df)
    prediction(test_df, pipeline, enc)
    
    
def load_data():
    train_df = pd.read_csv(r"C:\Users\bmatt\Desktop\Esercizi_ML\Esercizio credit Score\train.csv", dtype=({"Monthly_Balance":"str"}))
    test_df = pd.read_csv(r"C:\Users\bmatt\Desktop\Esercizi_ML\Esercizio credit Score\test.csv")
    train_df = train_df.drop(["Name", "ID","SSN"], axis=1)
    test_df = test_df.drop(["Name", "ID","SSN"], axis=1)
    return train_df, test_df


def train_features(train_df):
    train_df["Credit_History_Age"] = train_df["Credit_History_Age"].str.extract(r'(\d+)').astype(float)
    columns_to_num = ["Changed_Credit_Limit", "Age", "Credit_History_Age", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance", "Monthly_Inhand_Salary", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment"]
    for colonna in columns_to_num:
        train_df[colonna] = train_df[colonna].astype(str).str.replace("_", "", regex=False)
        train_df[colonna] = pd.to_numeric(train_df[colonna], errors='coerce')
        
    train_df = train_df.dropna(subset="Credit_Score")
    train_df_2 = train_df.drop("Credit_Score", axis=1)
    
    print("NAN Analysis: ")
    categorical_columns=[]
    for col in train_df_2.columns:
        if train_df_2[col].dtype == 'object':
            categorical_columns.append(col)
            print(f"{col}: {train_df_2[col].isna().sum()}")
            
    numeric_columns = []
    for col in train_df_2.columns:
        if train_df_2[col].dtype in ["float64", "int64"]:
            numeric_columns.append(col)
            print(f"{col}: {train_df_2[col].isna().sum()}")
            
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler(with_mean=False))])
    
    preprocessor = ColumnTransformer(transformers=[("categoric", cat_pipeline, categorical_columns),
                                                  ("num", num_pipeline, numeric_columns)])
    
    return preprocessor, train_df
 
    
def model(preprocessor, train_df):
    
    models = [RandomForestClassifier(),  
               LogisticRegression(max_iter=1000),
               SVC()]
    
    enc = LabelEncoder()
    
    y = enc.fit_transform(train_df["Credit_Score"])
    X = train_df.drop("Credit_Score", axis=1)
    
    print(f"Features shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    
    f1_macro_val = []
    for model in models:
        modello = Pipeline(steps=[("prepoc", preprocessor), ("model", model)])
        name_modello = type(model).__name__
        f1_macro = cross_val_score(modello, X, y, scoring="f1_macro", cv=5)
        print("Model: ", name_modello)
        print("Accuracy after 5-fold CV:", round(np.mean(f1_macro),2))
        f1_macro_val.append(f1_macro)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        # modello.fit(X_train, y_train)
        # y_pred = modello.predict(X_test)
        # print(classification_report(y_test, y_pred, target_names=enc.classes_))
    
    f1_macro_means = [np.mean(score) for score in f1_macro_val]
    best_index = np.argmax(f1_macro_means)
    model_best = models[best_index]
    pipeline = Pipeline(steps=[("prepoc", preprocessor), ("model", model_best)])
    pipeline.fit(X, y)
    return pipeline, enc
    
    
def test_features(test_df):
    test_df["Credit_History_Age"] = test_df["Credit_History_Age"].str.extract(r'(\d+)').astype(float)
    columns_to_num = ["Changed_Credit_Limit", "Age", "Credit_History_Age", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance", "Monthly_Inhand_Salary", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment"]
    for col in columns_to_num:
        test_df[col] = test_df[col].astype(str).str.replace("_", "", regex=False)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    return test_df
    
def prediction(test_df, pipeline, enc):
    print("Predictions on the test dataset: ")
    y_pred = pipeline.predict(test_df)
    y_pred = enc.inverse_transform(y_pred)
    print(pd.Series(y_pred).value_counts(normalize=True))
    
    
if __name__=="__main__":
    main()
