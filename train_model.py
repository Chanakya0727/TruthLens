import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading datasets...")
# Define paths
DATA_DIR = r"c:\Users\chana\OneDrive\Desktop\project\ai_fake_news\data"
MODEL_DIR = r"c:\Users\chana\OneDrive\Desktop\project\ai_fake_news\model"

fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv"))

# Add target labels
# 0 = Fake News, 1 = True News
fake_df["class"] = 0
true_df["class"] = 1

# Since the dataset might be huge (40k+ rows), let's ensure we sample it slightly if memory is a concern, 
# or just use all of it. For robust training, we combine everything.
df_merge = pd.concat([fake_df, true_df], axis=0)

# We are generally predicting based on the text of the news. We can also combine 'title' + 'text'.
df_merge['text'] = df_merge['title'] + " " + df_merge['text']

# Drop unnecessary columns (we only need 'text' and 'class')
df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle the dataset

print("Preprocessing text...")
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df['text'] = df['text'].apply(wordopt)

# Define X and Y
x = df['text']
y = df['class']

print("Splitting datasets and vectorizing...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("Starting TF-IDF Vectorization. This may take a minute...")
vectorization = TfidfVectorizer(max_df=0.7, max_features=50000)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

print("Training Logistic Regression Model...")
lr = LogisticRegression()
lr.fit(xv_train, y_train)

# Test accuracy
pred_lr = lr.predict(xv_test)
acc = accuracy_score(y_test, pred_lr)
print(f"Logistic Regression Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, pred_lr))

# Saving the model and vectorizer
print("Saving model and vectorizer...")
pickle.dump(lr, open(os.path.join(MODEL_DIR, "model.pkl"), 'wb'))
pickle.dump(vectorization, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), 'wb'))

print(f"Implementation successful! Model and vectorizer saved to {MODEL_DIR}")
