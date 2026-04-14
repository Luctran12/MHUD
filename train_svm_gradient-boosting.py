# import pandas as pd
# import time
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.pipeline import Pipeline

# # 1. Đọc và tiền xử lý dữ liệu
# df = pd.read_csv('processed_data.csv')
# df['subject'] = df['subject'].fillna('')
# df['message'] = df['message'].fillna('')
# df['full_text'] = df['subject'] + " " + df['message']

# X = df['full_text']
# y = df['label']
# # Tách riêng 2 nhóm
# df_spam = df[df['label'] == 1]
# df_ham = df[df['label'] == 0]

# # Giảm mẫu Spam xuống bằng với số lượng Ham
# df_spam_downsampled = df_spam.sample(n=len(df_ham), random_state=42)

# # Ghép lại thành tập dữ liệu mới cân bằng 50-50
# df_balanced = pd.concat([df_spam_downsampled, df_ham]).sample(frac=1, random_state=42) # Xáo trộn lại

# # Chia tập Train/Test (80/20)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 2. Khởi tạo 2 Pipelines
# svm_pipeline = Pipeline([
#     ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), )),
#     ('classifier', LinearSVC(class_weight='balanced',random_state=42))
# ])

# gb_pipeline = Pipeline([
#     ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2), )),
#     ('classifier', GradientBoostingClassifier(random_state=42))
# ])

# print("BẮT ĐẦU HUẤN LUYỆN VÀ SO SÁNH...\n")

# # 3. Huấn luyện và đánh giá SVM
# print("⏳ Đang huấn luyện SVM...")
# start_time_svm = time.time()
# svm_pipeline.fit(X_train, y_train)
# time_svm = time.time() - start_time_svm

# svm_pred = svm_pipeline.predict(X_test)
# acc_svm = accuracy_score(y_test, svm_pred)
# # Dùng average='weighted' để tính F1-score có xét đến tỷ lệ chênh lệch giữa lượng mail Ham và Spam
# f1_svm = f1_score(y_test, svm_pred, average='weighted') 
# print(f"✅ SVM Xong! Thời gian train: {time_svm:.2f} giây")

# # 4. Huấn luyện và đánh giá Gradient Boosting
# print("\n⏳ Đang huấn luyện Gradient Boosting (Sẽ mất thời gian)...")
# start_time_gb = time.time()
# gb_pipeline.fit(X_train, y_train)
# time_gb = time.time() - start_time_gb

# gb_pred = gb_pipeline.predict(X_test)
# acc_gb = accuracy_score(y_test, gb_pred)
# f1_gb = f1_score(y_test, gb_pred, average='weighted')
# print(f"✅ Gradient Boosting Xong! Thời gian train: {time_gb:.2f} giây")

# # 5. Lưu cả 2 mô hình ra ổ cứng
# print("\n💾 Đang xuất file mô hình...")
# joblib.dump(svm_pipeline, 'model_svm_spam.pkl')
# joblib.dump(gb_pipeline, 'model_gb_spam.pkl')
# print("✅ Đã lưu thành công 2 file: 'model_svm_spam.pkl' và 'model_gb_spam.pkl'")

# # 6. In bảng tổng kết đối chiếu
# print("\n" + "="*75)
# print("🏆 BẢNG TỔNG KẾT SO SÁNH MÔ HÌNH (ACCURACY & F1-SCORE)")
# print("="*75)
# print(f"{'Mô hình':<20} | {'Accuracy':<12} | {'F1-Score (Weighted)':<20} | {'Thời gian Train'}")
# print("-" * 75)
# print(f"{'1. Linear SVM':<20} | {acc_svm * 100:>11.2f}% | {f1_svm:>19.4f} | {time_svm:>10.2f} giây")
# print(f"{'2. Gradient Boosting':<20} | {acc_gb * 100:>11.2f}% | {f1_gb:>19.4f} | {time_gb:>10.2f} giây")
# print("="*75)

# model_filename = 'spam_classifier_model.pkl'
# joblib.dump(model, model_filename)

# print(f"✅ Đã lưu mô hình thành công vào file: {model_filename}")
#7. Thử nghiệm thực tế (Dự đoán một email mới)
# new_emails = [
#     "Hey Billy, trust me.. girls love bigger ones, website is http://ctmay.com", # Spam rõ ràng
#     "Hi team, let's have a meeting tomorrow at 10 AM to discuss the new project.", # Mail công việc bình thường
#     "Dear Customer,We have detected unusual activity in your account. If you do not verify immediately, your account will be locked within 24 hours.Please click the link below to verify:http://xacminh-taikhoan-secure123.comIf you fail to act, we are not responsible for any loss.Sincerely,Customer Support Team"
# ]

# print("\n--- THỬ DỰ ĐOÁN MAIL MỚI ---")
# predictions = model.predict(new_emails)
# for email, label in zip(new_emails, predictions):
#     status = "Spam" if label == 1 else "Bình thường"
#     print(f"[{status}] - {email[:60]}...")





import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# 1. Đọc và tiền xử lý dữ liệu
df = pd.read_csv('processed_data.csv')
df['subject'] = df['subject'].fillna('')
df['message'] = df['message'].fillna('')
df['full_text'] = df['subject'] + " " + df['message']

# Xử lý cân bằng dữ liệu 50-50
df_spam = df[df['label'] == 1]
df_ham = df[df['label'] == 0]
min_size = min(len(df_spam), len(df_ham))
df_balanced = pd.concat([
    df_spam.sample(n=min_size, random_state=42),
    df_ham.sample(n=min_size, random_state=42)
]).sample(frac=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['full_text'], df_balanced['label'], test_size=0.2, random_state=42
)

# 2. Khởi tạo Pipelines
svm_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,2))),
    ('classifier', LinearSVC(class_weight='balanced', random_state=42, dual=False))
])

gb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# 3. Huấn luyện
print("⏳ Đang huấn luyện SVM...")
svm_pipeline.fit(X_train, y_train)
svm_pred = svm_pipeline.predict(X_test)

print("⏳ Đang huấn luyện Gradient Boosting...")
gb_pipeline.fit(X_train, y_train)
gb_pred = gb_pipeline.predict(X_test)

# --- PHẦN VẼ ĐỒ THỊ ---

def plot_results(y_true, svm_preds, gb_preds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Confusion Matrix cho SVM
    cm_svm = confusion_matrix(y_true, svm_preds)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix: SVM')
    axes[0].set_xlabel('Dự đoán')
    axes[0].set_ylabel('Thực tế')

    #2. Confusion Matrix cho Gradient Boosting
    cm_gb = confusion_matrix(y_true, gb_preds)
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Confusion Matrix: Gradient Boosting')
    axes[1].set_xlabel('Dự đoán')
    axes[1].set_ylabel('Thực tế')

    # 3. So sánh Accuracy & F1
    metrics = {
        'Model': ['SVM', 'SVM', 'GB', 'GB'],
        'Score': [accuracy_score(y_true, svm_preds), f1_score(y_true, svm_preds, average='weighted'),
                  accuracy_score(y_true, gb_preds), f1_score(y_true, gb_preds, average='weighted')],
        'Metric': ['Accuracy', 'F1-Score', 'Accuracy', 'F1-Score']
    }
    df_metrics = pd.DataFrame(metrics)
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_metrics, ax=axes[2])
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title('So sánh hiệu suất mô hình')
    
    plt.tight_layout()
    plt.show()

# Gọi hàm vẽ
plot_results(y_test, svm_pred, gb_pred)

# Lưu mô hình tốt nhất (ví dụ SVM thường tốt hơn cho Text)
joblib.dump(svm_pipeline, 'spam_classifier_model.pkl')
print("\n✅ Đã lưu mô hình và hiển thị biểu đồ!")