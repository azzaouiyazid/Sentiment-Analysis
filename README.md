# Sentiment Analysis with Machine Learning

![Sentiment Analysis](sentiment-analysis.png)

Sentiment analysis, also known as opinion mining, is a fascinating field of Natural Language Processing (NLP) that focuses on determining the emotional tone behind a piece of text. In this project, we employ various machine learning algorithms to perform sentiment analysis on a dataset of textual reviews.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Cross-Validation](#cross-validation)
- [Results and Conclusions](#results-and-conclusions)

## Project Overview

The primary goal of this project is to develop a robust sentiment analysis model capable of classifying text into positive or negative sentiment. We leverage Python and several powerful libraries for this purpose. Below is a detailed explanation of the code and its components.

## Prerequisites

Before diving into the project, make sure you have the following prerequisites set up:

1. **Python Environment and Libraries:** You need a Python environment with the following libraries installed:
   - `scikit-learn` (sklearn): For machine learning models.
   - `pandas`: For data manipulation and analysis.
   - `numpy`: For numerical operations.
   - `matplotlib` and `seaborn`: For data visualization.
   - `nltk` (Natural Language Toolkit): For text preprocessing.
   - `spacy`: For more advanced text preprocessing.
   - `google.colab`: For Google Colab integration.

2. **Dataset:**
   - You'll need a CSV dataset for training and testing your sentiment analysis model. Ensure it has two columns: 'review' (textual data) and 'sentiment' (positive or negative sentiment labels). Due to the large size of the dataset (55MB), it's recommended to compress it into a zip file and include it in your repository.

   **Note:** Make sure to mention in the README that the dataset is compressed into a zip file in the repository.

## Data Preprocessing

1. **Data Loading:** We start by loading the dataset into a Pandas DataFrame. Ensure that your dataset path is correctly set in the code. For example:

   ```python
   with open('/content/drive/MyDrive/Dataset.csv') as file:
       articles = file
   df = pd.read_csv('/content/drive/MyDrive/Dataset.csv')
Data Preprocessing: Before feeding the data into machine learning models, we perform some essential preprocessing tasks:

Removing Stopwords: Common English words that don't add much meaning to the text are removed using NLTK's stopwords list.
python
Copy code
from nltk.corpus import stopwords
def remove_stopwords(X, y=None):
    stop_words = set(stopwords.words('english'))
    X_processed = []
    for doc in X:
        words = doc.split()
        words = [word for word in words if word not in stop_words]
        X_processed.append(' '.join(words))
    if y is not None:
        return X_processed, y
    else:
        return X_processed
Tokenization and Vectorization: We tokenize the text (split it into individual words) and convert it into numerical features using CountVectorizer from scikit-learn.
Model Building and Evaluation
Logistic Regression Model:

We build a sentiment analysis model using Logistic Regression with CountVectorizer for feature extraction.
python
Copy code
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),    # Vectorize the text data
    ('classifier', LogisticRegression(max_iter=86)), 
])
We fit the model on the training data and evaluate its performance on a test set using accuracy. We also visualize the results using a confusion matrix.
Random Forest Classifier Model:

A second sentiment analysis model is constructed using the Random Forest Classifier with CountVectorizer for feature extraction.
python
Copy code
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=185))
])
We assess the model's accuracy and display the confusion matrix to understand its performance better.
Precision-Recall Curve:

We generate a precision-recall curve to understand the model's precision and recall characteristics for different probability thresholds.
python
Copy code
y_pred_proba = pipeline.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])
Cross-Validation
To ensure robustness and generalization of the models, we employ 5-fold cross-validation for two different classifiers: Logistic Regression and Random Forest Classifier. We assess each classifier's performance using accuracy and standard deviation.

python
Copy code
from sklearn.model_selection import cross_val_score

# Create a list of classifiers to compare
classifiers = [
    LogisticRegression(),
    RandomForestClassifier(),
]

# Iterate over the classifiers and compute the cross-validation scores
for clf in classifiers:
    pipeline = Pipeline([
        ('stopwords', FunctionTransformer(remove_stopwords)),
        ('vectorizer', CountVectorizer()),    # Vectorize the text data
        ('classifier', clf), # Train a classifier
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    print(f'{clf.__class__.__name__} - Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
Results and Conclusions
After running the code, we obtained the following results:

Logistic Regression Model:

Accuracy: 0.8804897959183674
Confusion Matrix:
Negative Class:
True Negative: 87.32%
False Positive: 12.68%
Positive Class:
False Negative: 11.23%
True Positive: 88.77%
Random Forest Classifier Model:

Accuracy: 0.8558367346938776
Confusion Matrix:
Negative Class:
True Negative: 83.99%
False Positive: 16.01%
Positive Class:
False Negative: 12.83%
True Positive: 87.17%
The Logistic Regression model achieved higher accuracy compared to the Random Forest Classifier. However, the warning messages suggest that the Logistic Regression model may benefit from increasing the number of iterations or scaling the data. Further tuning and optimization of hyperparameters may improve its performance.

In conclusion, this project provides a foundation for sentiment analysis using machine learning techniques. By referencing the mentioned documentation and exploring alternative solver options, you can enhance the Logistic Regression model's performance. Additionally, you have the flexibility to fine-tune both models and adapt them to your specific use cases. Enjoy sentiment analysis on large textual datasets!
