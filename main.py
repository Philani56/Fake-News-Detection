import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Load datasets for fake and true news
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Assign class labels to the datasets
data_fake["class"] = 0  # Fake news labeled as 0
data_true['class'] = 1  # True news labeled as 1

# Prepare datasets for manual testing
data_fake_manual_testing = data_fake.tail(10).copy()  # Last 10 samples from fake news for manual testing
data_fake = data_fake.iloc[:-10]  # Remove last 10 samples from the fake news dataset

data_true_manual_testing = data_true.tail(10).copy()  # Last 10 samples from true news for manual testing
data_true = data_true.iloc[:-10]  # Remove last 10 samples from the true news dataset

# Combine the fake and true news datasets
data_merge = pd.concat([data_fake, data_true], axis=0).reset_index(drop=True)

# Drop unnecessary columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Text preprocessing function
def wordopt(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
    text = re.sub(r"\\W", " ", text)  # Remove non-word characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    return text

# Apply text preprocessing to the 'text' column
data['text'] = data['text'].apply(wordopt)

# Define features and labels
x = data['text']  # Feature: news text
y = data['class']  # Label: news class

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorization of text data using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()  # Initialize the TF-IDF Vectorizer
xv_train = vectorization.fit_transform(x_train)  # Fit and transform the training data
xv_test = vectorization.transform(x_test)  # Transform the test data

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()  # Initialize the Logistic Regression model
LR.fit(xv_train, y_train)  # Train the model

# Predictions and Evaluation for Logistic Regression
pred_lr = LR.predict(xv_test)  # Predict using the trained model
print(f"Accuracy: {LR.score(xv_test, y_test)}")  # Print accuracy
print(classification_report(y_test, pred_lr))  # Print classification report

# Decision Tree Classifier Model
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()  # Initialize the Decision Tree model
DT.fit(xv_train, y_train)  # Train the model

# Predictions and Evaluation for Decision Tree
pred_dt = DT.predict(xv_test)  # Predict using the trained model
print(f"Decision Tree Accuracy: {DT.score(xv_test,y_test)}")  # Print accuracy
print(classification_report(y_test, pred_dt))  # Print classification report

# Function to map output label to its text representation
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}  # Create a DataFrame with the new news text
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)  # Apply text preprocessing
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)  # Vectorize the new news text
    pred_LR = LR.predict(new_xv_test)  # Predict using Logistic Regression
    pred_DT = DT.predict(new_xv_test)  # Predict using Decision Tree

    # Print the predictions
    print("\n\nLR Prediction: {} \nDT Prediction: {}".format(output_lable(pred_LR[0]), output_lable(pred_DT[0])))

# Taking user input for manual testing
news = str(input("Enter the news text for testing: "))  # Get news text from the user
manual_testing(news)  # Perform manual testing
