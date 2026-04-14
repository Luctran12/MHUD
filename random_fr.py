# ===================== 1. IMPORT =====================
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

import joblib

nltk.download('stopwords')


# ===================== 2. LOAD DATA =====================
df = pd.read_csv('processed_data.csv')

df['subject'] = df['subject'].fillna('')
df['message'] = df['message'].fillna('')
df['text'] = df['subject'] + " " + df['message']

df = df[['label', 'text']]

df['label'] = df['label'].astype(str).str.strip().str.lower()

df['label_num'] = df['label'].map({
    'ham': 0,
    'spam': 1,
    '0': 0,
    '1': 1
})

df = df.dropna(subset=['label_num'])


# ===================== 🔥 2.1 SHOW SAMPLE DATA =====================
print("\n📌 SAMPLE DATA:")
print(df.sample(10))

print("\n📌 SPAM EXAMPLES:")
print(df[df['label_num'] == 1].sample(5))

print("\n📌 HAM EXAMPLES:")
print(df[df['label_num'] == 0].sample(5))


# ===================== 3. VISUALIZE =====================
print("\n📊 Label distribution:")
print(df['label_num'].value_counts())

df['label_num'].value_counts().plot(kind='bar')
plt.title("Spam vs Ham Distribution")
plt.show()


# ===================== 4. PREPROCESS =====================
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    
    # 🔥 giữ stopwords (để không mất tín hiệu spam)
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

corpus = df['text'].apply(preprocess_text)


# ===================== 5. VECTORIZE =====================
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(corpus)
y = df['label_num']


# ===================== 6. SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===================== 7. TRAIN =====================
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)

clf.fit(X_train, y_train)


# ===================== 🔥 7.1 CHECK OVERFITTING =====================
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print("\n🎯 Train Accuracy:", train_acc)
print("🎯 Test Accuracy:", test_acc)

plt.plot([train_acc, test_acc])
plt.xticks([0,1], ['Train','Test'])
plt.title("Overfitting Check")
plt.show()


# ===================== 8. EVALUATE =====================
y_pred = clf.predict(X_test)

print("\n===== METRICS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===================== 9. CONFUSION MATRIX =====================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()


# ===================== 🔥 10. FEATURE IMPORTANCE =====================
importances = clf.feature_importances_
feature_names = vectorizer.get_feature_names_out()

indices = np.argsort(importances)[-15:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top Important Words")
plt.show()


# ===================== 🔥 11. PCA VISUALIZATION =====================
X_dense = X.toarray()

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_dense)

plt.figure()
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, alpha=0.3)
plt.title("PCA Visualization (Spam vs Ham)")
plt.show()


# ===================== 12. SAVE =====================
joblib.dump(clf, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model & Vectorizer saved!")