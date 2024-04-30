import docx
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    preprocessed_text = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(preprocessed_text)

# Load training data
with open('trainingData.pkl', 'rb') as f:
    trainingData = pickle.load(f)

# Convert graph representations to text
corpus = []
categories = []
for record in trainingData:
    text = " ".join(record['graph'].nodes())
    category = record['category']
    corpus.append(text)
    categories.append(category)

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(corpus)

# Train KNN classifier
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, categories)

# Function to classify test document using KNN
def classify_with_knn(test_text):
    preprocessed_test_text = preprocess_text(test_text)
    X_test = vectorizer.transform([preprocessed_test_text])
    return knn_classifier.predict(X_test)[0]

# Load and classify test documents
test_folder_path = "testDocx"
true_labels = []
predicted_labels = []

for filename in os.listdir(test_folder_path):
    if filename.endswith(".docx"):
        doc = docx.Document(os.path.join(test_folder_path, filename))
        true_category = filename.split('_')[0]
        true_labels.append(true_category)
        test_text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        predicted_category = classify_with_knn(test_text)
        predicted_labels.append(predicted_category)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Generate confusion matrix
labels = sorted(set(true_labels))
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
