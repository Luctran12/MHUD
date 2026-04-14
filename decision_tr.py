# # ===================== 1. IMPORT =====================
# import string
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# # 🔥 ĐÃ THAY ĐỔI: Chuyển sang DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.decomposition import PCA

# import joblib

# nltk.download('stopwords')


# # ===================== 2. LOAD DATA =====================
# df = pd.read_csv('processed_data.csv')

# df['subject'] = df['subject'].fillna('')
# df['message'] = df['message'].fillna('')
# df['text'] = df['subject'] + " " + df['message']

# df = df[['label', 'text']]

# df['label'] = df['label'].astype(str).str.strip().str.lower()

# df['label_num'] = df['label'].map({
#     'ham': 0,
#     'spam': 1,
#     '0': 0,
#     '1': 1
# })

# df = df.dropna(subset=['label_num'])


# # ===================== 🔥 2.1 SHOW SAMPLE DATA =====================
# print("\n📌 SAMPLE DATA:")
# print(df.sample(10))

# # ===================== 3. VISUALIZE =====================
# print("\n📊 Label distribution:")
# print(df['label_num'].value_counts())

# df['label_num'].value_counts().plot(kind='bar')
# plt.title("Spam vs Ham Distribution")
# plt.show()


# # ===================== 4. PREPROCESS =====================
# stemmer = PorterStemmer()
# stopwords_set = set(stopwords.words('english'))

# def preprocess_text(text):
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     words = text.split()
    
#     # 🔥 giữ stopwords (để không mất tín hiệu spam)
#     words = [stemmer.stem(word) for word in words]
    
#     return ' '.join(words)

# corpus = df['text'].apply(preprocess_text)


# # ===================== 5. VECTORIZE =====================
# vectorizer = TfidfVectorizer(
#     max_features=3000,
#     ngram_range=(1, 2)
# )

# X = vectorizer.fit_transform(corpus)
# y = df['label_num']


# # ===================== 6. SPLIT =====================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )


# # ===================== 7. TRAIN (DECISION TREE) =====================
# # 🔥 ĐÃ THAY ĐỔI: Khởi tạo Decision Tree
# clf = DecisionTreeClassifier(
#     max_depth=30,             # Decision Tree cần giới hạn độ sâu để tránh overfitting
#     min_samples_split=5,
#     min_samples_leaf=2,
#     class_weight='balanced',  # Vẫn giữ balanced vì dữ liệu của bạn lệch
#     random_state=42
# )

# clf.fit(X_train, y_train)


# # ===================== 🔥 7.1 CHECK OVERFITTING =====================
# train_acc = clf.score(X_train, y_train)
# test_acc = clf.score(X_test, y_test)

# print("\n🎯 Train Accuracy:", train_acc)
# print("🎯 Test Accuracy:", test_acc)

# plt.plot([train_acc, test_acc])
# plt.xticks([0,1], ['Train','Test'])
# plt.title("Overfitting Check (Decision Tree)")
# plt.show()


# # ===================== 8. EVALUATE =====================
# y_pred = clf.predict(X_test)

# print("\n===== METRICS =====")
# print("Accuracy:", accuracy_score(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))


# # ===================== 9. CONFUSION MATRIX =====================
# cm = confusion_matrix(y_test, y_pred)
# plt.imshow(cm)
# plt.title("Confusion Matrix")
# plt.colorbar()
# plt.show()


# # ===================== 🔥 10. FEATURE IMPORTANCE =====================
# importances = clf.feature_importances_
# feature_names = vectorizer.get_feature_names_out()
# indices = np.argsort(importances)[-15:]

# plt.figure()
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
# plt.title("Top Important Words (Decision Tree)")
# plt.show()


# # ===================== 🔥 11. PCA VISUALIZATION =====================
# X_dense = X.toarray()
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X_dense)

# plt.figure()
# plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, alpha=0.3)
# plt.title("PCA Visualization")
# plt.show()


# # ===================== 12. SAVE =====================
# joblib.dump(clf, "decision_tr_model.pkl")
# joblib.dump(vectorizer, "vectorizer_decision_tr.pkl")

# print("\n✅ Model & Vectorizer (Decision Tree) saved!")

# ===================== 1. IMPORT =====================
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# 🔥 ĐÃ THAY ĐỔI: Chuyển từ RandomForest sang DecisionTree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# tải stopwords (chỉ chạy 1 lần)
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

print("\n📌 SAMPLE DATA:")
print(df.sample(5))

# ===================== 3. VISUALIZE =====================
print("\n📊 Label distribution:")
print(df['label_num'].value_counts())

df['label_num'].value_counts().plot(kind='bar')
plt.title("Spam vs Ham Distribution")
plt.show()

# ===================== 4. PREPROCESS =====================
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

technical_garbage = {
    'texthtml', 'quotedprint', 'contenttyp', 'contenttransferencod',
    'charset', 'boundary', 'mime', 'href', 'encoding', 'html', 'ascii'
}

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    words = text.split()

    cleaned_words = []
    for word in words:
        if word not in stopwords_set and word not in technical_garbage:
            word = stemmer.stem(word)
            if len(word) > 2:
                cleaned_words.append(word)

    return ' '.join(cleaned_words)

print("⏳ Đang xử lý dữ liệu...")
corpus = df['text'].apply(preprocess_text)
print("✅ Done preprocess")

# ===================== 5. VECTORIZE =====================
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

X = vectorizer.fit_transform(corpus)
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 6. TRAIN (DECISION TREE) =====================
# 🔥 ĐÃ THAY ĐỔI: Sử dụng DecisionTreeClassifier
clf = DecisionTreeClassifier(
    max_depth=30,             # Cây đơn lẻ thường cần sâu hơn 1 chút để bắt kịp logic
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # Giúp xử lý việc dữ liệu bị lệch (Spam > Ham)
    random_state=42
)

clf.fit(X_train, y_train)

print("\n🎯 Train Accuracy:", clf.score(X_train, y_train))
print("🎯 Test Accuracy :", clf.score(X_test, y_test))

# ===================== 7. EVALUATE =====================
y_pred = clf.predict(X_test)

print("\n===== METRICS (Decision Tree) =====")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Decision Tree)")
plt.show()

# ===================== 8. PCA =====================
# Lưu ý: PCA với dữ liệu thưa (Sparse) 3000 features có thể tốn RAM
X_dense = X.toarray()
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_dense)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, alpha=0.4, cmap='viridis')
plt.title("PCA Visualization (Spam vs Ham)")
plt.show()

# ===================== 9. SAVE =====================
joblib.dump(clf, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Saved Decision Tree model!")