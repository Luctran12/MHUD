import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocessing import preprocess

# ⏱️ START TOTAL TIME
total_start = time.time()
print("🚀 Đang khởi động tiến trình huấn luyện Logistic Regression (Final Version)...")

# LOAD DATA
print("📂 Đang tải dữ liệu từ CSV...")
df = pd.read_csv("data/processed_data.csv", encoding='latin-1').dropna(subset=['subject', 'message'])
df['text'] = df['subject'] + " " + df['message']

print("🧹 Đang tiền xử lý văn bản (Clean text & Stopwords)...")
df['clean_text'] = df['text'].apply(preprocess)

# ===============================
# ⏱️ TF-IDF TIME
# ===============================
print(f"🧬 Đang trích xuất đặc trưng TF-IDF (N-grams: 1-2, Sublinear TF: ON)...")
start_vec = time.time()

# CẢI TIẾN: sublinear_tf giúp kìm hãm sự áp đảo của các từ khóa đơn lẻ
vectorizer = TfidfVectorizer(
    max_features=10000, 
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,                    # 🔥 bỏ từ hiếm
    max_df=0.9                   # 🔥 bỏ từ quá phổ biến
)
X = vectorizer.fit_transform(df['clean_text'])
vectorize_time = time.time() - start_vec

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, df['label'], test_size=0.2, random_state=42
)

# ===============================
# ⏱️ TRAIN TIME
# ===============================
print("🧠 Đang huấn luyện Logistic Regression (Balanced Weight)...")
start_train = time.time()

# CẢI TIẾN: class_weight='balanced' giúp xử lý tốt hơn nếu dữ liệu bị lệch
model = LogisticRegression(
    max_iter=1000, 
    C=0.8, 
    class_weight='balanced', 
    solver='lbfgs'
)
model.fit(X_train, y_train)
train_time = time.time() - start_train

# ===============================
# ⏱️ PREDICT TIME
# ===============================
start_pred = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - start_pred

# SAVE
print("💾 Đang lưu mô hình và bộ vector hóa...")
pickle.dump(model, open("models/logistic_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer_log.pkl", "wb"))

total_time = time.time() - total_start

print("✅ Logistic model saved!")
print("\n⏱️ THỜI GIAN:")
print(f"Vectorize time: {vectorize_time:.4f} seconds")
print(f"Train time     : {train_time:.4f} seconds")
print(f"Predict time   : {predict_time:.4f} seconds")
print(f"Total time     : {total_time:.4f} seconds")