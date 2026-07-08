import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Read the dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only the required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels into numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Convert messages into numerical values
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

# Target values
y = data['label']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model trained successfully!")