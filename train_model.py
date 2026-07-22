import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Dataset load
data = pd.read_csv("spam.csv")


# Dataset columns check
print(data.head())


# Select columns
# spam.csv lo columns: v1 = label, v2 = message ani assume chestunnam

data = data[['v1','v2']]


# Labels convert
data['v1'] = data['v1'].map({
    'ham':0,
    'spam':1
})


# Input and output
X = data['v2']
y = data['v1']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english'
)


X_train_vector = vectorizer.fit_transform(X_train)

X_test_vector = vectorizer.transform(X_test)



# Model training
model = MultinomialNB()

model.fit(
    X_train_vector,
    y_train
)



# Accuracy
prediction = model.predict(X_test_vector)

accuracy = accuracy_score(
    y_test,
    prediction
)


print("Model Accuracy:", accuracy*100)



# Create model folder
if not os.path.exists("model"):
    os.makedirs("model")



# Save model
joblib.dump(
    model,
    "model/spam_model.pkl"
)


# Save vectorizer
joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)


print("Model and Vectorizer saved successfully")