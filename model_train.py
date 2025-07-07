import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

print("🔄 Downloading stopwords...")
nltk.download('stopwords')

# Preprocess Function
ps = PorterStemmer()
def preprocess(text):
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

print("📥 Loading dataset...")
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['processed'] = df['message'].apply(preprocess)

print("🔢 Vectorizing...")
cv = CountVectorizer()
X = cv.fit_transform(df['processed']).toarray()
y = df['label']

print("🧠 Training model...")
model = MultinomialNB()
model.fit(X, y)

print("💾 Saving model and vectorizer...")
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("✅ Model training complete.")
