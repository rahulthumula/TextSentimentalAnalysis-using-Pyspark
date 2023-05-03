# Sentiment Analysis using Spark ML

This project demonstrates how to perform sentiment analysis on a dataset of tweets using PySpark. The dataset used in this project contains around 1.6 million tweets, and each tweet is labeled either as positive or negative. The goal of the project is to train a machine learning model that can accurately classify a tweet as positive or negative based on its text.

## Getting Started

The project was developed in Google Colab, so there's no need to install any dependencies or set up any environment. However, you can also run it locally on your machine by installing PySpark, Jupyter Notebook, and other required dependencies. The project requires Python 3.6 or higher.

## Project Workflow

### Dataset
The dataset used for this project is a preprocessed version of the Sentiment140 dataset. It contains 1.6 million tweets with sentiment labels (positive or negative).
### Preprocessing
Before training the machine learning model, the tweets are preprocessed to clean and tokenize the text. The preprocessing steps include:

Removing URLs, mentions, and special characters from the tweets
Converting the text to lowercase
Tokenizing the text into words
Removing stop words and words with a length less than 3

### Data Cleaning:
The dataset is cleaned using regular expressions to remove special characters, punctuations, numbers, and URLs.

### Tokenization: 
The text is split into words and stored in a new column.

### Stop Words Removal: 
Stop words like "the", "a", "an", etc., are removed from the text.

### N-grams Creation: 
The text is split into n-grams to capture the context of the text.

### TF-IDF Vectorization: 
The n-grams are converted into vectors using the TF-IDF algorithm.

## Training and Testing: 
The data is split into training and testing sets, and a Logistic Regression model is trained on the training data.

## Evaluation: 
The model is evaluated using accuracy, F1-score, confusion matrix, and ROC-AUC.

## Usage
To use this project, you'll need to follow these steps:

Clone this repository to your local machine or open it in Google Colab.

Download a Twitter dataset in CSV format.

Update the file path in the code to point to your dataset.

Run the code cells in order to perform sentiment analysis on your dataset.

## Requirements
Python 3.6 or higher
Apache Spark 2.4.0 or higher
PySpark 2.4.0 or higher
Jupyter Notebook or Google Colab
pandas, numpy, matplotlib, seaborn, and nltk Python packages.


## Conclusion
The trained model achieves an accuracy of around 80% and an F1 score of around 0.79 on the test dataset, which is a decent performance considering the size and nature of the dataset. The project demonstrates how to use PySpark and MLlib to train a machine learning model for sentiment analysis and provides a starting point for further exploration and experimentation.



