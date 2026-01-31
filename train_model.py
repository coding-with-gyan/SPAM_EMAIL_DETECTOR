import pandas as pd
import re
import pickle
import nltk
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ================= TEXT CLEANING =================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove numbers & symbols
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ================= LOAD DATA =================
df = pd.read_csv("spam.csv", sep='\t', names=['label', 'message'])

# Fix message column (remove NaN)
df['message'] = df['message'].fillna('').astype(str)

# Fix label column
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove broken rows
df = df.dropna(subset=['label', 'message'])

# Clean text
df['message'] = df['message'].apply(clean_text)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ================= TF-IDF =================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ================= CLASS IMBALANCE FIX =================
classes = np.array([0, 1])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
weights = {0: class_weights[0], 1: class_weights[1]}

# ================= MODEL =================
model = LogisticRegression(class_weight=weights, max_iter=1000)
model.fit(X_train_vec, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test_vec)

print("\n🎯 Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ================= SAVE FILES =================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n🔥 PRO SPAM MODEL TRAINED & SAVED SUCCESSFULLY 🔥")
