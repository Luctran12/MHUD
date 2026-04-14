# import joblib
# import pandas as pd
# import re
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# nltk.download('stopwords')

# # load model
# try:
#     clf = joblib.load("spam_model.pkl")
#     vectorizer = joblib.load("vectorizer.pkl")
#     print("✅ Load model thành công!")
# except Exception as e:
#     print("❌ Lỗi load model:", e)
#     exit()

# # preprocess
# stemmer = PorterStemmer()
# stopwords_set = set(stopwords.words('english'))

# technical_garbage = {
#     'texthtml','quotedprint','contenttyp','contenttransferencod',
#     'charset','boundary','mime','href','encoding','html','ascii'
# }

# def preprocess_text(text):
#     text = str(text).lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\d+', '', text)
#     words = text.split()

#     return ' '.join([
#         stemmer.stem(w)
#         for w in words
#         if w not in stopwords_set
#         and w not in technical_garbage
#         and len(w) > 2
#     ])

# # ===================== 30 MẪU =====================
# test_samples = [
#     "Are we still on for the football match tonight?", "Please find the attached invoice for last month's services.",
#     "Can you pick up some milk on your way home?", "The meeting has been rescheduled to 3 PM in Room 402.",
#     "Thanks for the help yesterday, I really appreciate it.", "Did you see the new movie trailer I sent you?",
#     "I'll be home late, don't wait up for dinner.", "Can we jump on a quick call to discuss the budget?",
#     "Happy Anniversary! Looking forward to our dinner tonight.", "Your doctor's appointment is confirmed for Tuesday at 10 AM.",
#     "Don't forget to submit your assignment by midnight.", "Hey, do you have the contact number for the plumber?",
#     "The weather looks great for a hike this weekend.", "I'm heading to the gym, catch you later!",
#     "Just checking in to see how your first day at work went.",
#     "WINNER! Your phone number was selected to receive a $5000 prize. Call 090xxxx now!",
#     "Urgent: Your bank account will be suspended. Verify your details at http://bit.ly/fake-link",
#     "Get a personal loan up to $50,000 with 0% interest. Apply inside!",
#     "Congratulations! You've been chosen for a free luxury cruise. Reply YES to claim.",
#     "Make $2000 a week working from home! No experience needed. Join now.",
#     "Final Reminder: Your Netflix subscription has failed. Update payment here.",
#     "Hot girls in your area want to meet you! Click to see profiles.",
#     "You have (1) uncollected package. Pay the shipping fee at this link to receive it.",
#     "Invest in Bitcoin now and triple your money in 24 hours! Guaranteed.",
#     "Claim your free gift card worth $100. Limited time offer, act fast!",
#     "Your Apple ID is locked for security reasons. Log in here to unlock.",
#     "Lose 10kg in 1 week with this secret pill! Order today for 50% off.",
#     "Earn extra income by testing products. Sign up for our mailing list.",
#     "PRIVATE: We have a business proposal for you. Contact us for details.",
#     "Verify your identity to receive your tax refund of $450.75 immediately."
# ]

# true_labels = [0]*15 + [1]*15

# # ===================== PREDICT =====================
# processed_samples = [preprocess_text(t) for t in test_samples]
# X_new = vectorizer.transform(processed_samples)

# predictions = clf.predict(X_new)
# probs = clf.predict_proba(X_new)

# # ===================== HIỂN THỊ =====================
# results_df = pd.DataFrame({
#     'Nội dung': [t[:50] + "..." for t in test_samples],
#     'Thực tế': ['Ham' if l==0 else 'Spam' for l in true_labels],
#     'Dự đoán': ['Ham' if p==0 else 'Spam' for p in predictions],
#     'Độ tin cậy (%)': [round(max(pr)*100, 2) for pr in probs]
# })

# print("\n===== KẾT QUẢ 30 MẪU =====")
# print(results_df.to_string(index=False))

# # ===================== THỐNG KÊ =====================
# correct = (predictions == true_labels).sum()
# print(f"\n✅ Model đoán đúng {correct}/{len(test_samples)} mẫu")
# print(f"🎯 Accuracy: {round(correct/len(test_samples)*100,2)}%")

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# ===================== 1. PREPROCESS (UPGRADED) =====================
import nltk
nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

technical_garbage = {
    'texthtml', 'quotedprint', 'contenttyp', 'contenttransferencod',
    'charset', 'boundary', 'mime', 'href', 'encoding', 'html', 'ascii'
}

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    
    cleaned_words = []
    for word in words:
        if word not in stopwords_set and word not in technical_garbage:
            stemmed_word = stemmer.stem(word)
            if len(stemmed_word) > 2:
                cleaned_words.append(stemmed_word)
    return ' '.join(cleaned_words)

# ===================== 2. LOAD MODEL & VECTORIZER =====================
try:
    clf = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Đã load Model và Vectorizer thành công!")
except:
    print("❌ Lỗi: Không tìm thấy file .pkl. Hãy chạy file train trước!")
    exit()

# ===================== 3. CREATE 20 TEST CASES =====================
test_data = [
    # 10 SPAM
    {"text": "CONGRATULATIONS! You've won a $1000 Walmart gift card. Click here to claim now!!!", "label": 1},
    {"text": "URGENT: Your account access is restricted. Please verify your identity at bit.ly/fake-link", "label": 1},
    {"text": "Free entry into our £100 weekly draw. Text WIN to 80082 to receive entry question.", "label": 1},
    {"text": "Get cheap viagra and cialis! No prescription needed. Best prices online.", "label": 1},
    {"text": "Investment opportunity: Double your Bitcoin in 24 hours. Guaranteed returns!", "label": 1},
    {"text": "Dear Customer, your invoice is overdue. Please see the attached file for payment details.", "label": 1},
    {"text": "Work from home! Earn $5000 a month just by typing. No experience required.", "label": 1},
    {"text": "Final notice: You have an unclaimed tax refund of $450. Click to process.", "label": 1},
    {"text": "Meet hot singles in your area tonight. Registration is 100% free!", "label": 1},
    {"text": "Your computer is infected with a virus. Call our tech support line immediately.", "label": 1},
    
    # 10 HAM
    {"text": "Hey, are we still meeting for lunch at 12:30 today? Let me know.", "label": 0},
    {"text": "The project deadline has been moved to next Friday. Please update your schedule.", "label": 0},
    {"text": "Hi Mom, I'll be home late tonight. Don't wait for me for dinner.", "label": 0},
    {"text": "Can you please send me the meeting minutes from yesterday's session?", "label": 0},
    {"text": "Happy Birthday! Hope you have a wonderful day with your family.", "label": 0},
    {"text": "Thank you for your application. We will review it and get back to you soon.", "label": 0},
    {"text": "The lecture on Quantum Physics has been canceled due to the professor's illness.", "label": 0},
    {"text": "I've attached the draft of the contract. Let me know if you have any comments.", "label": 0},
    {"text": "Don't forget to buy milk and eggs on your way back home.", "label": 0},
    {"text": "Regarding our conversation, I've scheduled the interview for Monday at 10 AM.", "label": 0}
]

df_test = pd.DataFrame(test_data)

# ===================== 4. PREDICT WITH 90% THRESHOLD =====================
df_test['clean_text'] = df_test['text'].apply(preprocess_text)
X_test_custom = vectorizer.transform(df_test['clean_text'])

# Lấy xác suất lớp Spam
probabilities = clf.predict_proba(X_test_custom)[:, 1]
df_test['prob_spam'] = probabilities

# Logic mới: Chỉ là spam nếu prob >= 0.9
df_test['prediction'] = (df_test['prob_spam'] >= 0.9).astype(int)

# ===================== 5. VISUALIZATION (GIỮ NGUYÊN FORM CŨ) =====================
print("\n" + "="*30)
print("📊 KẾT QUẢ KIỂM THỬ THỰC TẾ (Threshold 90%)")
print("="*30)

for i, row in df_test.iterrows():
    status = "✅ ĐÚNG" if row['label'] == row['prediction'] else "❌ SAI"
    label_str = "SPAM" if row['label'] == 1 else "HAM"
    pred_str = "SPAM" if row['prediction'] == 1 else "HAM"
    print(f"Case {i+1:02d} | Thật: {label_str:4s} | Dự đoán: {pred_str:4s} | Xác suất Spam: {row['prob_spam']:.2f} | {status}")

# 1. Vẽ Confusion Matrix
cm = confusion_matrix(df_test['label'], df_test['prediction'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix on Custom Test Cases')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 2. Vẽ biểu đồ xác suất (Probability Bar Chart)
df_test['color'] = df_test['label'].map({0: 'green', 1: 'red'})
plt.figure(figsize=(12, 6))
plt.bar(range(1, 21), df_test['prob_spam'], color=df_test['color'], alpha=0.7)
plt.axhline(y=0.9, color='blue', linestyle='--', label='Threshold 90%') # Thêm đường line 90%
plt.title('Spam Probability for each Case (Green=Ham, Red=Spam labels)')
plt.ylabel('Spam Probability')
plt.xlabel('Case Number')
plt.xticks(range(1, 21))
plt.legend()
plt.tight_layout()
plt.show()

# 3. In Classification Report
print("\n📝 BÁO CÁO CHI TIẾT:")
print(classification_report(df_test['label'], df_test['prediction'], target_names=['Ham', 'Spam']))
print(f"Tổng Accuracy: {accuracy_score(df_test['label'], df_test['prediction']) * 100:.2f}%")