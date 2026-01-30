import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv")
df['label'] = df['v1'].map({'ham':0,'spam':1})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['v2'])

model = MultinomialNB()
model.fit(X, df['label'])

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("MODEL READY")
