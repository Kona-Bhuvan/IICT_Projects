# 🛡️ AI-Driven Phishing Email Detection System — Web Interface

An end-to-end machine learning web application built with **Flask**, **Natural Language Processing (NLP)**, and **scikit-learn**. This application serves real-time predictions to classify emails as **PHISHING** or **LEGITIMATE** using a hybrid feature extraction pipeline (Structural Metadata + TF-IDF Lexical Features + spaCy Word Embeddings).

---

## 📌 Project Architecture & Overview

The backend classifier is powered by a **Multi-Layer Perceptron (MLP) Neural Network** trained on the `CEAS_08` dataset, achieving an overall **Accuracy of 99.32%** and an **F1-Score of 0.9939**.

### Directory Structure
```text
phishing_webapp/
├── app.py                  # Flask web application routing & inference engine
├── README.md               # Setup guide and project documentation
├── requirements.txt        # System dependencies & required Python libraries
├── static/
│   └── style.css           # Custom UI stylesheet
├── templates/
│   └── index.html          # Web dashboard interface template
└── model_artifacts/        # Serialized pipeline artifacts copied from training
    ├── best_model.joblib   # Trained MLP Neural Network model weights
    ├── scaler.joblib       # Fitted StandardScaler object
    ├── tfidf.joblib        # Fitted TfidfVectorizer object
    └── model_info.txt      # Model metadata log and feature shape specs
```

---

## 🚀 Setup & Execution Guide

### Step 1: Export Model Artifacts from Notebook

In the training notebook (`IICT_Project_2.ipynb`), run the `export_artifacts_cell.py` script once model training completes. This creates the `model_artifacts/` directory containing:

* `best_model.joblib`

* `scaler.joblib`

* `tfidf.joblib`

* `model_info.txt`


### Step 2: Copy Artifacts to Web App

Ensure the exported `model_artifacts/` folder is placed directly inside the `phishing_webapp/` project directory as shown in the file tree above.

### Step 3: Install Dependencies

Navigate into `phishing_webapp/` and install the required packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

(Note: NLTK stopwords and tokenization datasets will be downloaded automatically on the first execution of `app.py`.)

### Step 4: Run the Application

Start the Flask local development server:

```bash
python app.py

```

Open your web browser and navigate to:

```text
http://127.0.0.1:5000

```

---

## 🔍 How to Test

1. Paste an email **Subject** and/or **Body** into the web dashboard interface.


2. Click **🔍 Analyze Message**.


3. View the classification verdict (**PHISHING** vs **LEGITIMATE**), confidence scores, and extracted metadata indicators (URL count, urgency triggers, uppercase ratios, exclamation marks).



---
