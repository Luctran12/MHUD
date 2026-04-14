import pandas as pd
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.preprocessing import preprocess

# ⏱️ TOTAL TIME
total_start = time.time()
print("🚀 Đang khởi động tiến trình huấn luyện KNN...")

# LOAD
print("📂 Đang tải và tiền xử lý dữ liệu...")
df = pd.read_csv("data/processed_data.csv", encoding='latin-1')

# FIX NULL (QUAN TRỌNG)
df = df.dropna(subset=['subject', 'message'])

df['text'] = df['subject'] + " " + df['message']
df['clean_text'] = df['text'].apply(preprocess)

X_text = df['clean_text']
y = df['label']

# ===============================
# ⏱️ TF-IDF (GIỮ MỨC 5000 ĐỂ TỐI ƯU CHO KNN)
# ===============================
print("🧬 Đang trích xuất đặc trưng TF-IDF (Max features: 5000)...")
start = time.time()

# Sử dụng Unigram (ngram_range=(1,1)) và 5000 features cho KNN
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

vectorize_time = time.time() - start

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# ⏱️ SCALING
# ===============================
print("⚖️ Đang thực hiện chuẩn hóa dữ liệu (Scaling)...")
start = time.time()

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)

scale_time = time.time() - start

# ===============================
# ⏱️ TRAIN
# ===============================
print("👥 Đang huấn luyện mô hình KNN (n=5, metric=cosine)...")
start = time.time()

knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(X_train_scaled, y_train)

train_time = time.time() - start

# ===============================
# ⏱️ PREDICT
# ===============================
print("🔮 Đang thực hiện dự đoán thử nghiệm trên tập Test...")
X_test_scaled = scaler.transform(X_test)

start = time.time()
y_pred = knn.predict(X_test_scaled)
predict_time = time.time() - start

# ===============================
# 💾 LƯU ASSETS (QUAN TRỌNG: LƯU THÊM VECTORIZER_KNN)
# ===============================
print("💾 Đang lưu mô hình và bộ vector hóa riêng cho KNN...")
pickle.dump(knn, open("models/knn_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
# Lưu vectorizer này với tên riêng để không đè lên bản 10k của Logistic
pickle.dump(vectorizer, open("models/vectorizer_knn.pkl", "wb")) 

# TOTAL
total_time = time.time() - total_start

# ===============================
# 📊 PRINT
# ===============================
print("✅ KNN model and Vectorizer_knn saved!")

print("\n⏱️ THỜI GIAN THỰC THI:")
print(f"Vectorize time: {vectorize_time:.4f} seconds")
print(f"Scaling time  : {scale_time:.4f} seconds")
print(f"Train time    : {train_time:.4f} seconds")
print(f"Predict time  : {predict_time:.4f} seconds")
print(f"Total time    : {total_time:.4f} seconds")