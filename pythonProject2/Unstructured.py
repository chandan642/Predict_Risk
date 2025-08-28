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

structured_data = {
    'loan_id': range(1, 21),
    'credit_score': [720, 680, 750, 580, 800, 620, 710, 550, 760, 690,
                     730, 600, 780, 520, 650, 700, 640, 790, 590, 740],
    'loan_amount': [15000, 50000, 25000, 80000, 10000, 60000, 30000, 95000,
                    12000, 45000, 20000, 85000, 18000, 100000, 35000, 40000,
                    55000, 22000, 90000, 28000],
    'risk_label': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                   0, 1, 0, 1, 0, 0, 1, 0, 1, 0]  # 1 = High Risk, 0 = Low Risk
}
structured_df = pd.DataFrame(structured_data)
unstructured_df = pd.read_csv("unstructured_data.csv")
df = structured_df.merge(unstructured_df, on="loan_id", how="left")

print("--- Sample of the merged DataFrame ---")
print(df.head())
print("\n" + "="*50 + "\n")

try:
    nltk.data.find('F:\\PredictiveModel\\pythonProject2\\.venv\\nltk_data')
except LookupError:
    nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    return analyzer.polarity_scores(str(text))['compound']

df['sentiment_score'] = df['unstructured_data'].apply(get_sentiment)

tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(df['unstructured_data'].astype(str)).toarray()
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_feature_names)

print("--- Sample of TF-IDF Features ---")
print(tfidf_df.head())
print("\n" + "="*50 + "\n")

X = pd.concat([df[['credit_score', 'loan_amount', 'sentiment_score']], tfidf_df], axis=1)
y = df['risk_label']
  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\n" + "="*50 + "\n")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("--- Model Evaluation ---")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

print("--- Top 10 Most Important Features ---")
print(sorted_importances.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_importances.head(10), y=sorted_importances.head(10).index)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
