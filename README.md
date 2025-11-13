# Kindle User Review Sentiment Analysis
## Project Description
This project aims to classify Kindle user reviews as either positive or negative sentiment. It involves data loading, extensive text preprocessing, feature engineering using various techniques (CountVectorizer, TF-IDF, Word2Vec), and training several machine learning models (Multinomial Naive Bayes, Logistic Regression, Random Forest, Support Vector Machine) to evaluate their performance on this binary classification task.

## Setup and Installation
To run this notebook, you need to install the following Python libraries:

!pip install gensim nltk scikit-learn pandas numpy matplotlib seaborn
Ensure you have NLTK data downloaded. You can run the following in a code cell:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
## Data
The project uses two datasets: train.txt and test.txt, which are expected to be located in /content/sample_data/Kindle_user_review/. These files contain Kindle user reviews, each prefixed with a sentiment label (__label__1 for negative, __label__2 for positive).

## Workflow
### Data Loading:
The train.txt and test.txt files are loaded into pandas DataFrames.
### Data Preprocessing:
Labels are extracted and stored in a 'Ratings' column.
The review text is cleaned by removing punctuation, non-alphabetic characters, converting to lowercase, removing stopwords, and performing lemmatization.
## Feature Engineering:
### CountVectorizer (BOW): 
Converts text data into a matrix of token counts.
### TF-IDF Vectorizer:
Converts text data into a matrix of TF-IDF features.
### Word2Vec:
Generates word embeddings, and then sentence embeddings by averaging word vectors for each review.
### Model Training:
The following machine learning models are trained and evaluated:  
Multinomial Naive Bayes (for CountVectorizer and TF-IDF)  
Logistic Regression  
Random Forest Classifier  
Support Vector Machine (SVC)
### Evaluation:
Models are evaluated using accuracy and a detailed classification report (precision, recall, f1-score).
## Performance Highlights
### Using CountVectorizer (BOW)
MultinomialNB Accuracy: 0.847  
LogisticRegression Accuracy: 0.869  
### Using TF-IDF
MultinomialNB Accuracy: 0.848  
LogisticRegression Accuracy: 0.878  
### Using Word2Vec
LogisticRegression Accuracy: 0.509  
(Note: This suggests Word2Vec with simple averaging might not be capturing sentiment effectively with Logistic Regression in this specific setup, or further fine-tuning/different models are needed for Word2Vec embeddings).
