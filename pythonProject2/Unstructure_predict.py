import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Synthesize a Dummy Dataset ---
# In a real-world scenario, this data would be sourced from internal systems
# and external APIs. We create a small, representative dataset for this example.

data = {
    'loan_id': range(1, 21),
    'credit_score': [720, 680, 750, 580, 800, 620, 710, 550, 760, 690,
                     730, 600, 780, 520, 650, 700, 640, 790, 590, 740],
    'loan_amount': [15000, 50000, 25000, 80000, 10000, 60000, 30000, 95000,
                    12000, 45000, 20000, 85000, 18000, 100000, 35000, 40000,
                    55000, 22000, 90000, 28000],
    'unstructured_data': [
        "Customer reported a recent job loss and is struggling with payments.",
        "Positive review from a third-party analyst. No issues reported.",
        "Successful product launch. Company is expanding into new markets.",
        "Multiple news reports of supply chain disruptions and production delays.",
        "Excellent market reputation. Strong financial outlook.",
        "Internal incident report notes minor compliance issues.",
        "Client is a key partner in a large government contract. No red flags.",
        "Company facing legal action for environmental violations.",
        "News article highlights strong Q3 earnings. Solid growth expected.",
        "Customer inquiry about early loan repayment options. Good sign.",
        "Minor customer complaint regarding service speed, quickly resolved.",
        "Employee turnover is high. Negative sentiment on social media.",
        "Company acquired a competitor. Industry analysts are optimistic.",
        "News reports indicate significant layoffs and restructuring plans.",
        "Customer's business is stable and has a history of on-time payments.",
        "Public statement from CEO about positive future growth.",
        "Internal report mentions a data breach. Potential operational risk.",
        "New partnership announced. Market response is very positive.",
        "Customer's business is in a declining industry. High competition.",
        "No new information. Standard business operations."
    ],
    'risk_label': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                   0, 1, 0, 1, 0, 0, 1, 0, 1, 0] # 1 = High Risk, 0 = Low Risk
}

df = pd.DataFrame(data)

print("--- Sample of the created DataFrame ---")
print(df.head())
print("\n" + "="*50 + "\n")
print("Printing the NLTK BASE PATH")
print(nltk.data.path)

# --- 2. Feature Engineering from Unstructured Data (NLP) ---

# Download necessary NLTK component for sentiment analysis
try:
   # nltk.data.find('sentiment/vader_lexicon.zip')
   nltk.data.find('F:\\PredictiveModel\\pythonProject2\\.venv\\nltk_data')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

# Create a new feature for sentiment score
df['sentiment_score'] = df['unstructured_data'].apply(get_sentiment)

# Use TF-IDF to vectorize the text data
# TF-IDF gives higher weight to words that are important to a specific document
# but not too common across all documents.
tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(df['unstructured_data']).toarray()
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert the TF-IDF features to a DataFrame
tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_feature_names)

print("--- Sample of TF-IDF Features ---")
print(tfidf_df.head())
print("\n" + "="*50 + "\n")

# --- 3. Data Integration and Preparation for Modeling ---

# Combine the structured features and the new NLP features
X = pd.concat([df[['credit_score', 'loan_amount', 'sentiment_score']], tfidf_df], axis=1)
y = df['risk_label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\n" + "="*50 + "\n")

# --- 4. Model Development and Training ---

# Initialize and train a Random Forest Classifier
# This model is great for showcasing feature importance.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Prediction and Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test)

print("--- Model Evaluation ---")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Feature Importance Analysis ---
# This is crucial for understanding how the model works.
# It will show the influence of both structured and unstructured data features.

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

print("--- Top 10 Most Important Features ---")
print(sorted_importances.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_importances.head(10), y=sorted_importances.head(10).index)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Conclusion on the code:
# This example demonstrates how you can turn free-form text into numerical features
# that a machine learning model can use to predict risk. The feature importance
# plot visually highlights which aspects of the data, including our NLP-derived
# sentiment and TF-IDF features, are most influential in the model's decision-making process.
# This approach adds a new dimension to risk analysis that goes beyond
# traditional structured data.
