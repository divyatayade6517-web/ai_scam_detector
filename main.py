
import pandas as pd

# Read the SMS dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display the first five messages
print(data.head())

# Display the total number of messages
print("\nTotal messages:", len(data))

# Count normal and spam messages
print("\nMessage category counts:")
print(data["label"].value_counts())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())


# Separate input messages and output labels
X = data["message"]
y = data["label"]

# Display sample input messages
print("\nSample messages:")
print(X.head())

# Display their correct labels
print("\nSample labels:")
print(y.head())

# Display the size of our dataset
print("\nTotal messages:", len(X))

from sklearn.model_selection import train_test_split

# Split messages into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

# Create the TF-IDF converter
vectorizer = TfidfVectorizer()

# Learn vocabulary and convert training messages into numbers
X_train_tfidf = vectorizer.fit_transform(X_train)

# Convert testing messages using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)


from sklearn.naive_bayes import MultinomialNB

# Create the AI model
model = MultinomialNB()

# Train the model using messages and their correct labels
model.fit(X_train_tfidf, y_train)

print("\nAI model trained successfully!")

from sklearn.metrics import accuracy_score, classification_report

# Predict the categories of testing messages
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel accuracy:", round(accuracy * 100, 2), "%")

# Display precision, recall and F1-score
print("\nClassification report:")
print(classification_report(y_test, y_pred))

import joblib
import os

# Create a folder to store the trained model
os.makedirs("model", exist_ok=True)

# Save the trained AI model and TF-IDF converter
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nAI model and vectorizer saved successfully!")
