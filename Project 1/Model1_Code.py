#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('WELFake_Dataset.csv')
print(df.describe())
print(df.isna().sum())
df.head()


# In[2]:


import re
import string
import nltk
from nltk.corpus import stopwords

df = df.dropna(subset=['text']).reset_index(drop=True)
df['title'] = df['title'].fillna('')
df['full_text'] = df['title'] + " " + df['text']

nltk.download('stopwords', quiet=True)
stopWords = set(stopwords.words('english'))

def clean_and_tokenization(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    translator = str.maketrans('', '', string.punctuation + string.digits)
    text = text.translate(translator)
    # tokenization
    tokens = text.split()
    cleanedTokens = [word for word in tokens if word not in stopWords]

    return " ".join(cleanedTokens)

df['cleanedText'] = df['full_text'].apply(clean_and_tokenization)
df['cleanedText'].head(6)


# In[3]:


from sklearnex import patch_sklearn
patch_sklearn()

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
import time

# Bag of Words [Top most frequent words]
print("Bag of Words... ", end = "")
start = time.time()
bow_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000)
X_bow = bow_vectorizer.fit_transform(df['cleanedText'])
print(f"Completed; Time Taken: {(time.time() - start):.3}")

# TF-IDF [Distribution of word importance weights]
print("TF-IDF... ", end = "")
start = time.time()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleanedText'])
print(f"Completed; Time Taken: {(time.time() - start):.3}")

# Embeddings
print("Embeddings... ", end = "")
start = time.time()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = embedding_model.encode(df['cleanedText'].tolist(), batch_size=64)
print(f"Completed; Time Taken: {(time.time() - start):.3}")

# Hybrid
print("Hybrid... ", end = "")
start = time.time()
X_embeddings_sparse = sp.csr_matrix(X_embeddings)
X_hybrid = sp.hstack([sp.csr_matrix(X_tfidf), X_embeddings_sparse])
print(f"Completed; Time Taken: {(time.time() - start):.3}")


# In[4]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

y = df['label'].values

feature_dictionary = {
    "Bag-of-Words (N-grams)": X_bow,
    "TF-IDF (N-grams)": X_tfidf,
    "Embeddings": X_embeddings,
    "Hybrid Matrix (TF-IDF + Embeddings)": X_hybrid
}

results = []

for feature_name, X_matrix in feature_dictionary.items():
    print(f"--- Training models using: {feature_name} ---")

    X_train, X_test, y_train, y_test = train_test_split(X_matrix, y, test_size=0.25, random_state=42)

    model_dictionary = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=2, n_jobs=-1),
        "Logistic Regression": LogisticRegression(C=20.0, random_state=42, max_iter=1500),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, n_jobs=-1),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
    }

    for model_name, model_object in model_dictionary.items():
        model_object.fit(X_train, y_train)
        predictions = model_object.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f" [✓] {model_name} completed. Accuracy: {accuracy:.3%}")
        results.append({
            "Feature Representation": feature_name,
            "ML Model": model_name,
            "Accuracy": accuracy
        })
    print()

print("=== Best Performance Configuration ===")
df_summary = pd.DataFrame(results)
best_run = df_summary.loc[df_summary['Accuracy'].idxmax()]
print(f"Top Feature Setup: {best_run['Feature Representation']}")
print(f"Top Machine Learning Model: {best_run['ML Model']}")
print(f"Maximum Accuracy Achieved: {best_run['Accuracy']:.3%}")

