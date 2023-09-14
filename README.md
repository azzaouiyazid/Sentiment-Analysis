# Sentiment Analysis with Machine Learning



## Project Overview

Sentiment analysis, also known as opinion mining, is a valuable Natural Language Processing (NLP) application that aims to determine the emotional tone behind text. In this project, we employ various machine learning algorithms to perform sentiment analysis on a dataset of textual reviews.

## Prerequisites

Before getting started, ensure you have the following prerequisites:

- **Python Environment and Libraries:** You'll need a Python environment with the following libraries installed:
   - `scikit-learn` (sklearn): For machine learning models.
   - `pandas`: For data manipulation and analysis.
   - `numpy`: For numerical operations.
   - `matplotlib` and `seaborn`: For data visualization.
   - `nltk` (Natural Language Toolkit): For text preprocessing.
   - `spacy`: For more advanced text preprocessing.
   - `google.colab`: For Google Colab integration.

- **Dataset:**
   - Prepare a CSV dataset for training and testing your sentiment analysis model. Ensure it has two columns: 'review' (textual data) and 'sentiment' (positive or negative sentiment labels). Due to the large size of the dataset (55MB), consider compressing the zip file and include it in your repository.

   **Note:** Mention in the README that the dataset is compressed into a zip file in the repository.

## Data Preprocessing

1. **Data Loading:** Start by loading the dataset into a Pandas DataFrame. You can do this as follows:

   ```python
   with open('/content/drive/MyDrive/Dataset.csv') as file:
       articles = file
   df = pd.read_csv('/content/drive/MyDrive/Dataset.csv')

- **Data Preprocessing:** Before feeding the data into machine learning models, perform essential preprocessing tasks:

   - Removing Stopwords: Eliminate common English words that don't contribute significantly to the text's meaning using NLTK's stopwords list.

   - Tokenization and Vectorization: Tokenize the text (split it into individual words) and convert it into numerical features using CountVectorizer from scikit-learn.
 
## Model Building and Evaluation

- **Logistic Regression Model**: Build a sentiment analysis model using Logistic Regression. Here's how you can set up the model:

      ```python
       from sklearn.pipeline import Pipeline
       from sklearn.linear_model import LogisticRegression

       pipeline = Pipeline([
       ('vectorizer', CountVectorizer()),    # Vectorize the text data
       ('classifier', LogisticRegression(max_iter=86)), 
       ])

    - Fit the model on the training data and evaluate its performance on a test set using accuracy. Visualize the results using a confusion matrix.

2. **Random Forest Classifier Model**: Create a sentiment analysis model using the Random Forest Classifier with CountVectorizer for feature extraction:

      ```python
       from sklearn.ensemble import RandomForestClassifier
       pipeline = Pipeline([
       ('vectorizer', CountVectorizer()),
       ('classifier', RandomForestClassifier(n_estimators=185))
       ])

 - Assess the model's accuracy and display the confusion matrix to better understand its performance.

3.**Precision-Recall Curve**: Generate a precision-recall curve to analyze the model's precision and recall characteristics for different probability thresholds.


## Cross-Validation

To ensure the models' robustness and generalization, employ 5-fold cross-validation for both Logistic Regression and Random Forest Classifier models. Evaluate each classifier's performance using accuracy and standard deviation.

## Results and conclusion 

After running the code, we obtained the following results:

 - Logistic Regression Model:
     - Accuracy: 0.8804897959183674
 - Random Forest Classifier Model:
     - Accuracy: 0.8558367346938776
  
The Logistic Regression model outperformed the Random Forest Classifier in terms of accuracy. However, the warning messages suggest that the Logistic Regression model may benefit from increasing the number of iterations or scaling the data. Further tuning and optimization of hyperparameters may improve its performance.

In conclusion, this project provides a solid foundation for sentiment analysis using machine learning techniques. By referencing the provided documentation and exploring alternative solver options, you can enhance the Logistic Regression model's performance. Additionally, you have the flexibility to fine-tune both models and adapt them to your specific use cases. Enjoy sentiment analysis on large textual datasets!
   
