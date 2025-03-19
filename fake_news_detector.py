import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import pickle

# Step 1: Load the dataset
df = pd.read_csv('train.csv')
print("Dataset Loaded Successfully")

# Step 2: Data Description & Preprocessing
print("Dataset Info:\n", df.info())
print("Class Distribution:\n", df['label'].value_counts())

# Handling missing values
df = df.dropna(subset=['text', 'label'])

# Visualizing class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'], palette='coolwarm')
plt.title("Class Distribution: Real vs Fake News")
plt.show()
plt.close()

# Step 3: Split Data into Features and Labels
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Text Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 5: Model Selection & Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
plt.close()
# ROC Curve
y_prob = model.predict_proba(X_test_tfidf)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Step 7: Save Model & Vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Function to classify new news articles
def predict_news(text, threshold=0.6):
    text_tfidf = tfidf.transform([text])
    prob = model.predict_proba(text_tfidf)[0]
    print(f"Prediction Probabilities: [Real: {prob[0]:.4f}, Fake: {prob[1]:.4f}]")
    
    if prob[1] > threshold:
        return "Fake News"
    else:
        return "Real News"

# User input for prediction
if __name__ == "__main__":
    news_input = input("Enter a news article to classify: ")
    prediction = predict_news(news_input, threshold=0.8)
    print(f'Prediction: {prediction}')
