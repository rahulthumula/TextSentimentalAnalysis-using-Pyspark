# TextSentimentalAnalysis-using-Pysprak
Sentiment Analysis on Twitter Data
This is a project for performing sentiment analysis on Twitter data. The project is implemented in Python and uses Apache Spark as the distributed computing engine.

Requirements
Python 3.x
Apache Spark 3.x
PySpark
Jupyter Notebook or Google Colab
Dataset
The dataset used for this project is a preprocessed version of the Sentiment140 dataset. It contains 1.6 million tweets with sentiment labels (positive or negative).

Approach
The sentiment analysis is performed using a logistic regression model with trigram features. The model is trained on a portion of the dataset and then evaluated on the remaining portion. The evaluation metrics include accuracy, F1 score, ROC-AUC, and confusion matrix.

Files
sentiment_analysis.ipynb: Jupyter notebook containing the project code.
clean_tweet.csv: Preprocessed dataset file.
Usage
Download or clone the repository.
Open sentiment_analysis.ipynb in Jupyter Notebook or Google Colab.
Run the code cells to train the model and evaluate its performance.
Credits
This project was completed by [Your Name] as a part of [Course Name/Project Name].






