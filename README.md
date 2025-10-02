

# 🎬 IMDb Movie Review Sentiment Analysis

![IMDb Logo](https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg)

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#data-preprocessing)
6. [TF-IDF Vectorization](#tf-idf-vectorization)
7. [Machine Learning Models](#machine-learning-models)
8. [Model Evaluation](#model-evaluation)
9. [Saving the Model](#saving-the-model)
10. [Streamlit Web App](#streamlit-web-app)
11. [Installation](#installation)
12. [Usage](#usage)
13. [Folder Structure](#folder-structure)
14. [Future Improvements](#future-improvements)
15. [Credits](#credits)

---

## **Project Overview**

This project is a **Natural Language Processing (NLP)** application that analyzes **IMDb movie reviews** and predicts their **sentiment**—either **Positive** or **Negative**.

It combines:

* **TF-IDF (Term Frequency – Inverse Document Frequency)** for feature extraction
* **Machine Learning models** (Logistic Regression, Naive Bayes, SVM)
* A **Streamlit web app** for real-time sentiment prediction

This project demonstrates **text preprocessing, ML model training, and deployment**, making it suitable for CV/portfolio.

---

## **Problem Statement**

IMDb receives thousands of reviews daily. Manually analyzing sentiment is **time-consuming and inefficient**.

**Goal:** Automatically predict the sentiment of a movie review (positive or negative) using ML and NLP.

---

## **Dataset**

* **Source:** [IMDb Movie Review Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Size:** 50,000 reviews (balanced: 25,000 positive, 25,000 negative)
* **Columns:**

  * `review`: Movie review text
  * `sentiment`: Label (`positive` or `negative`)

**Notes:**

* Dataset is clean and balanced, ideal for supervised learning.

---

## **Exploratory Data Analysis (EDA)**

EDA helps understand patterns and characteristics of the dataset:

1. **Review Lengths:** Most reviews are between 50–300 words.
2. **Word Frequency:** Positive reviews frequently use words like "excellent", "amazing"; negative reviews use "boring", "waste".
3. **Visualization:** WordClouds help identify commonly used words.
4. **Sentiment Distribution:** Dataset is perfectly balanced.

**Purpose:** Ensures model learns patterns and avoids bias.

---

## **Data Preprocessing**

Preprocessing is critical for text-based ML:

1. Convert text to **lowercase**
2. Remove **HTML tags**
3. Remove **punctuation**
4. Remove **stopwords** (common words like “the”, “is”)
5. Perform **lemmatization** (convert `loved → love`)

This reduces noise and ensures the model focuses on **meaningful words**.

---

## **TF-IDF Vectorization**

* Converts text into **numerical features** suitable for ML models
* Highlights **important words** while down-weighting common ones
* Used **unigrams + bigrams** (`ngram_range=(1,2)`)
* `max_features=5000` to limit dimensionality

---

## **Machine Learning Models**

Multiple supervised ML models were trained and compared:

1. **Logistic Regression** – Best performing
2. **Naive Bayes** – Simple probabilistic model
3. **Support Vector Machine (SVM)** – Works well for text classification
4. **Random Forest (Optional)** – Ensemble method

**Training & Testing:**

* Split dataset: 80% training, 20% testing
* Evaluated with **accuracy, precision, recall, F1-score**

---

## **Model Evaluation**

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.89     | 0.88      | 0.90   | 0.89     |
| Naive Bayes         | 0.86     | 0.85      | 0.87   | 0.86     |
| SVM                 | 0.88     | 0.87      | 0.89   | 0.88     |

**Takeaway:** Logistic Regression chosen as **best model** for deployment.

---

## **Saving the Model**

* Saved using **Joblib**:

```python
joblib.dump(best_model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

* Ensures **model + TF-IDF vectorizer** can be reused in app for **real-time predictions**.

---

## **Streamlit Web App**

Features:

* User inputs **any movie review**
* Text is **preprocessed and transformed using TF-IDF**
* **Prediction displayed**: Positive/Negative
* **Confidence bar** shows probability
* Glowing boxes and emojis for interactive UI
* Vanta.js 3D animated background + custom CSS

**Code highlights:**

```python
import streamlit as st
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Preprocess function
def preprocess(text):
    # lowercase, remove punctuation, stopwords, lemmatize
    ...

# Input & prediction
user_input = st.text_area("Enter Review")
if st.button("Analyze Sentiment"):
    clean_input = preprocess(user_input)
    prediction = model.predict(tfidf.transform([clean_input]))
    st.success(f"Predicted Sentiment: {prediction[0]}")
```

---

## **Installation**

1. Clone the repo:

```bash
git clone <repo_url>
```

2. Navigate to project folder:

```bash
cd imdb-sentiment-analysis
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Usage**

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open the browser link provided by Streamlit
3. Enter any movie review and click **Analyze Sentiment**
4. See the predicted sentiment with **confidence**

---

## **Folder Structure**

```
imdb-sentiment-analysis/
│
├─ app.py                     # Streamlit app
├─ sentiment_model.pkl        # Trained ML model
├─ tfidf_vectorizer.pkl       # TF-IDF vectorizer
├─ requirements.txt           # All dependencies
├─ README.md                  # Project documentation
└─ dataset/
   └─ imdb_reviews.csv        # IMDb dataset (optional)
```

---

## **Future Improvements**

* Include **neutral sentiment** classification
* Add **word cloud visualization** in app
* Include **multi-language support**
* Deploy app on **Streamlit Cloud / Heroku** for public access

---

## **Credits**

* Dataset: [IMDb Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Streamlit: [https://streamlit.io/](https://streamlit.io/)
* NLP & ML: scikit-learn, NLTK
* UI Design: Custom CSS + Vanta.js

---

