# #Heart Disease Project AI4ALL, August 2025
# # Author: Natalie Hicks
# #Logistic Regression using sklearn

import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

#Load data from csv files
df = pd.read_csv(Path('extracted_features_df.csv'))

X_train = pd.read_csv(Path('X_train.csv'))
X_test = pd.read_csv(Path('X_test.csv'))
y_train = pd.read_csv(Path('y_train.csv'))
y_test = pd.read_csv(Path('y_test.csv'))

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.drop(columns=y_train.columns[0])
y_test = y_test.drop(columns=y_test.columns[0])

# Train multinomial logistic regression model
model = MultiOutputClassifier(
    LogisticRegression(solver='saga', max_iter=10000, random_state=42, class_weight='balanced')
)
model.fit(X_train, y_train)

# Evaluation of accuracy. Where results mean:
#Recall: The percentage of actual patients with a disease that were correctly identified
#F1-score: The balanced average of precision and recall (harmonic mean)
#support: The number of actual occurrences of each class in your test set
#Hamming Loss: Average number of label errors per instance
#Exact match accuracy: Where all labels have to match to be classified correctly
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=['AS', 'AR', 'MR', 'MS', 'N'], zero_division=0))
print(f"\nHamming Loss: {hamming_loss(y_test, y_pred):.4f}")
print(f"Exact Match Accuracy: {accuracy_score(y_test, y_pred):.4f}")

