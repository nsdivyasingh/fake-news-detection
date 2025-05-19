import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os

# Load both the true and fake news data
true_data = pd.read_csv(r'C:\Users\navee\OneDrive\Documents\GitHub\fake-news-detection/fake-news-detection\data\True.csv')
fake_data = pd.read_csv(r'C:\Users\navee\OneDrive\Documents\GitHub\fake-news-detection\fake-news-detection\data\Fake.csv')

print(true_data.head())
print(fake_data.head())

# Label the datasets
true_data['label'] = 1  # 1 for real news
fake_data['label'] = 0  # 0 for fake news

# Combine the datasets into one
data = pd.concat([true_data, fake_data], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords and punctuation
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and w not in string.punctuation]
    return ' '.join(words)

# Apply preprocessing to the text column
data['text'] = data['text'].apply(preprocess_text)

# Check the result of preprocessing
print(data['text'].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Transform the text data into TF-IDF features
X = vectorizer.fit_transform(data['text'])

# Extract the labels
y = data['label']

# Check the shape of the transformed data
print(X.shape)

from sklearn.model_selection import train_test_split

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

print(os.getcwd())  # This should print the path to the directory containing the script and the data folder.
