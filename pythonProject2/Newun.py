from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Assume 'df_encoded' is your final structured dataframe.
# Let's create a hypothetical unstructured column
df_encoded['customer_review'] = [
    "Great customer service and quick approval.",
    "Very slow process, not happy with the bank.",
    "Quick and efficient, highly recommend.",
    "The application was a bit confusing.",
    "Excellent service, smooth and fast approval.",
    "Had some issues with the documents, but it was resolved.",
    "Terrible experience, would not apply again.",
    "The best bank for a personal loan.",
    "Had a bad experience with the agent.",
    "Very easy to apply, happy with the loan.",
    "Prompt response and a hassle-free process.",
    "The interest rates are very high.",
    "Smooth and effortless, good service.",
    "Friendly staff and a fast approval.",
    "Took a long time to get approved."
] * (len(df_encoded) // 15) # Example: repeat to match df_encoded size
# Fill the rest if needed
remainder = len(df_encoded) % 15
if remainder > 0:
    df_encoded['customer_review'].iloc[-(remainder):] = [
        "Good experience.", "Fast approval."
    ] * (remainder // 2 + remainder % 2)

# --- NLP Preprocessing ---
# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

# Fit and transform the customer_review text data
tfidf_features = tfidf_vectorizer.fit_transform(df_encoded['customer_review'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.index = df_encoded.index # Align indices

# --- Combine Structured and Unstructured Data ---
# Drop the original text column and concatenate the new TF-IDF features
df_combined = pd.concat([df_encoded.drop('customer_review', axis=1), tfidf_df], axis=1)

# --- Run the XGBoost Model with Combined Data ---
y = df_combined['Approved_Flag']
x = df_combined.drop(['Approved_Flag'], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
xgb_classifier.fit(x_train, y_train)

y_pred = xgb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'\nAccuracy with combined structured and unstructured data: {accuracy:.2f}')
