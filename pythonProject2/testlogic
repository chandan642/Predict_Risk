import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# --- Data Ingestion ---
Internal_B = pd.read_excel("F:\Credit_Score_Modeling\Internal_Bank_Dataset.xlsx\Internal_Bank_Dataset.xlsx")
External_B = pd.read_excel("F:\Credit_Score_Modeling\External_Cibil_Dataset.xlsx\External_Cibil_Dataset.xlsx")

df1 = Internal_B.copy()
df2 = External_B.copy()

# Add a placeholder for unstructured data
# In a real-world scenario, you would load this from a file or database
unstructured_data = {
    'PROSPECTID': df1['PROSPECTID'].sample(frac=1).tolist(), # Example IDs
    'text': ['The applicant has a strong financial background.',
             'This borrower has a history of late payments.',
             'Current employment status is unstable.',
             'Income is high and stable.',
             'Multiple inquiries for new credit in the last 6 months.'] * (len(df1) // 5 + 1)
}
unstructured_df = pd.DataFrame(unstructured_data).head(len(df1))

# --- Data Merging & Cleaning ---
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]
columns_to_be_removed = [i for i in df2.columns if df2.loc[df2[i] == -99999].shape[0] > 10000]
df2 = df2.drop(columns_to_be_removed, axis=1)
for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]

# Merge all three dataframes
df = pd.merge(df1, df2, how='inner', on='PROSPECTID')
df = pd.merge(df, unstructured_df, how='inner', on='PROSPECTID')
# Handle potential missing text values by filling with an empty string
df['text'] = df['text'].fillna('')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

# Fit and transform the text data to a sparse matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Create a DataFrame from the sparse matrix
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Reset index to ensure alignment before concatenation
df = df.reset_index(drop=True)

# Concatenate the original dataframe with the new TF-IDF features
df_combined = pd.concat([df, tfidf_df], axis=1)# Handle potential missing text values by filling with an empty string
df['text'] = df['text'].fillna('')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

# Fit and transform the text data to a sparse matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Create a DataFrame from the sparse matrix
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Reset index to ensure alignment before concatenation
df = df.reset_index(drop=True)

# Concatenate the original dataframe with the new TF-IDF features
df_combined = pd.concat([df, tfidf_df], axis=1)
# --- Feature Selection on Combined Data ---
# List all numerical columns, including the new TF-IDF features
numeric_columns = [col for col in df_combined.columns if df_combined[col].dtype != 'object' and col not in ['PROSPECTID', 'Approved_Flag']]

# VIF sequentially check (same logic, just on the new dataframe and columns)
vif_data = df_combined[numeric_columns].copy()
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index += 1
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)

# Check Anova for columns_to_be_kept (same logic)
from scipy.stats import f_oneway
columns_to_be_kept_numerical = []
for i in columns_to_be_kept:
    a = list(df_combined[i])
    b = list(df_combined['Approved_Flag'])
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']
    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)

# Listing all final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df_final = df_combined[features + ['Approved_Flag']]

# --- Label Encoding (same as before) ---
df_final.loc[df_final['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df_final.loc[df_final['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df_final.loc[df_final['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df_final.loc[df_final['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df_final.loc[df_final['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df_final.loc[df_final['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1
df_final.loc[df_final['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3
df_final['EDUCATION'] = df_final['EDUCATION'].astype(int)
df_encoded = pd.get_dummies(df_final, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

# --- Modeling (Random Forest, XGBoost, Decision Tree) ---
y = df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')

# XGBoost
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
x_train, x_test, y_train_xgb, y_test_xgb = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
xgb_classifier.fit(x_train, y_train_xgb)
y_pred_xgb = xgb_classifier.predict(x_test)
accuracy_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb:.2f}')

# Decision Tree
dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
