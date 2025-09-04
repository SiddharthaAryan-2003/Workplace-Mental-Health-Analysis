# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# loading the dataset
data = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Desktop\classroom ppts\4th sem\ML\ML projectsurvey.csv')

# preprocess the data
features = ['family_history', 'work_interfere', 'no_employees', 'remote_work']
target = 'treatment'

# clean and encode categorical variables
le = LabelEncoder()
for column in features + [target]:
    data[column] = data[column].fillna('Unknown')
    data[column] = le.fit_transform(data[column])

# select features and target
X = data[features]
y = data[target]

# spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# model makes predictions baseed on the traning
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix - Treatment Seeking Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))