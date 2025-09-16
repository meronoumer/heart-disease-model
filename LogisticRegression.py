#Heart Disease Project AI4ALL, July 2025
# Author: Natalie Hicks
#Logistic Regression using sklearn
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

# Read data from path, using my path rn so change before running
df = pd.read_csv("extracted_features_df.csv")
# Define non-feature columns
non_feature_cols = [
    'Unnamed: 0', 'patient_id_x', 'file_key', 'audio_filename_base',
    'Age', 'Gender', 'Smoker', 'Lives', 'AS', 'AR', 'MR', 'MS', 'N'
]

# Get all feature columns
feature_cols = [col for col in df.columns if col not in non_feature_cols]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
feature_names = X.columns.tolist()

# Prepare multi-label targets
y = []
for _, row in df.iterrows():
    diseases = []
    for disease in ['AS', 'AR', 'MR', 'MS']:
        if row[disease] == 1:
            diseases.append(disease)
    y.append(diseases if diseases else ['N'])

mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y)

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multinomial logistic regression model
model = MultiOutputClassifier(
    LogisticRegression(solver='lbfgs', max_iter=10000, random_state=42, class_weight='balanced')
)
model.fit(X_train, y_train)

# Evaluation of accuracy (Not as accurate as needed currently). Where results mean:
#Recall: The percentage of actual patients with a disease that were correctly identified
#F1-score: The balanced average of precision and recall (harmonic mean)
#support: The number of actual occurrences of each class in your test set
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
print(f"\nHamming Loss: {hamming_loss(y_test, y_pred):.4f}")
print(f"Exact Match Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# Interactive Prediction Section (Was using for testing right now but is not finalized to account for everything we need)
#

filename = 'logistic_regression_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(model, file)
