#Heart Disease Project AI4ALL, July 2025
# Author: Natalie Hicks
#Logistic Regression using sklearn

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

# Read data from path, using my path rn so change before running
df = pd.read_csv(Path('C:/Users/natal/Downloads/extracted_features_df.csv'))

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
def predict_from_features():
    print("\n" + "=" * 50)
    print("DISEASE PREDICTION FROM AUDIO FEATURES")
    print("=" * 50)
    print(f"\nPlease enter all {len(feature_names)} feature values at once, separated by commas.")
    print("Example: 12.5, 0.04, 1500, -5.2, ... (and so on for all features)\n")

    while True:
        try:
            input_str = input("Enter all feature values: ")
            input_values = [float(x.strip()) for x in input_str.split(',')]

            if len(input_values) != len(feature_names):
                print(f"Error: Expected {len(feature_names)} values, got {len(input_values)}")
                continue

            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Make prediction
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)

            # Convert to disease names
            predicted_diseases = mlb.inverse_transform(prediction)
            diseases_list = predicted_diseases[0] if len(predicted_diseases[0]) > 0 else ['Normal (N)']

            # Get probabilities
            disease_probs = {disease: f"{proba[i][0][1] * 100:.1f}%" for i, disease in enumerate(mlb.classes_)}

            # Display results
            print("\n" + "=" * 50)
            print("PREDICTION RESULTS:")
            print(f"Predicted Conditions: {', '.join(diseases_list)}")
            print("\nProbability Estimates:")
            for disease, prob in disease_probs.items():
                print(f"{disease}: {prob}")
            print("=" * 50)
            break

        except ValueError:
            print("Invalid input. Please enter numbers only, separated by commas.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")




# Run interactive prediction for user
while True:
    predict_from_features()
    if input("\nPredict another sample? (y/n): ").lower() != 'y':
        print("Prediction session ended.")
        break

