# Credit Score Classification - Documentation

# Description:
This project aims to identify the best machine learning model for predicting a person's credit score based on a set of features. The analysis is based on two large datasets downloaded from Kaggle (https://www.kaggle.com/datasets/parisrohan/credit-score-classification). 
The main objective was to explore effective strategies for handling missing values and "dirty" data.

# Python Version: 
This project uses 3.11 Python.

# Problem Statement
As a data scientist at a global finance company, your task is to build an intelligent system that can classify individuals into appropriate credit score categories based on their credit-related information. The system should be capable of processing raw financial data, performing preprocessing, and applying machine learning models to predict the credit score class.

# Dataset
Two datasets are used:
_train.csv_: Used for training and validating the model.
_test.csv_: Used for making predictions using the trained model.

Removed Columns:
Name, ID, and SSN are dropped due to being identifiers and not useful for modeling.


# Project Structure

# 1. main()
Entry point of the program. It orchestrates the full pipeline:
- Loading data
- Preprocessing training data
- Training and selecting the best model
- Preprocessing test data
- Making predictions and exporting the final csv file.

# 2. load_data()
Loads the training and test datasets from CSV files. Drops non-essential columns (Name, ID, SSN, Payment_Behaviour) and returns cleaned DataFrames.

# 3. train_features(train_df)
Prepares the training dataset for modeling by:
- Cleaning and converting numerical columns from strings
- Handling missing and out of range values
- Identifying categorical and numerical features.
- Creating features and label (encoded)
  
Creates preprocessing pipelines:
- Categorical Pipeline: Uses SimpleImputer with a constant value and OneHotEncoder.
- Numerical Pipeline: Uses SimpleImputer (mean strategy) and StandardScaler.

Divides the dataset between training and test.

_Returns:_
- A fitted ColumnTransformer preprocessor
- Features and labels
- Encoder fitted
- Train and Test sets

# 4. model(preprocessor, train_df)
- Trains and selects the best classification model after a GridSearch.
- Models considered: RandomForestClassifier, LogisticRegression, XGBClassifier, GradientBoostingClassifier().
- Selects the model with the highest accuracy.
- Saves the best best model.

_Returns A fitted pipeline (preprocessing + model)_

# 5. validation((pipeline, y_holdout, X_holdout, enc):
- Print the classification report on the test set.

# 6. test_features(test_df)
Preprocesses the test dataset in the same way as the training dataset:
- Converts applicable string-based columns to numeric format
- Handles missing and malformed values

_Returns the cleaned test dataset._

# 7. prediction(test_df, pipeline, enc)
- Uses the trained pipeline to make predictions on the test data.
- Transforms the model's numeric predictions back to label format using the label encoder.
- Displays the distribution of predicted credit scores.
- Creates and saves a file csv with final dataset that includes the predictions.

# Technologies Used
pandas, numpy for data manipulation
scikit-learn for modeling and preprocessing

# Notes
Ensure that the file paths to the CSVs (train.csv and test.csv) are correctly specified.
The model training includes simple hyperparameter-free versions of each classifier for performance benchmarking.
Missing values and inconsistent formatting are handled gracefully via pipelines.
