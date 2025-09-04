import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the dataset
data = pd.read_csv('survey.csv')

# Select features and target
selected_features = [
    'Age', 'Gender', 'self_employed', 'family_history',
    'work_interfere', 'no_employees', 'remote_work',
    'benefits', 'care_options', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'mental_vs_physical'
]
target = 'treatment'

# Drop rows with missing target
data = data.dropna(subset=[target])

# Clean age and gender
data = data[(data['Age'] >= 16) & (data['Age'] <= 100)]
data['Gender'] = data['Gender'].fillna('').str.lower().str.strip()
data['Gender'] = data['Gender'].replace({
    'male': 'male', 'm': 'male', 'male-ish': 'male', 'man': 'male',
    'female': 'female', 'f': 'female', 'woman': 'female', 'femake': 'female',
    'trans-female': 'male', 'trans woman': 'male', 'transgender': 'trans',
    'trans male': 'female', 'trans man': 'female', 'agender': 'nonbinary',
    'non-binary': 'nonbinary', 'genderqueer': 'nonbinary', 'genderfluid': 'nonbinary'
})
data['Gender'] = data['Gender'].where(data['Gender'].isin(['male', 'female', 'trans', 'nonbinary']), 'other')

# Feature engineering: Create new features
data['has_benefits'] = data['benefits'].apply(lambda x: 1 if x in ['Yes', "Don't know"] else 0)
data['has_care_options'] = data['care_options'].apply(lambda x: 1 if x == 'Yes' else 0)
data['has_anonymity'] = data['anonymity'].apply(lambda x: 1 if x == 'Yes' else 0)
data['work_interfere_binary'] = data['work_interfere'].apply(lambda x: 1 if x in ['Often', 'Sometimes'] else 0)

# Update selected features with new engineered features
selected_features += ['has_benefits', 'has_care_options', 'has_anonymity', 'work_interfere_binary']

# Separate features and target
X = data[selected_features]
y = data[target].replace({'Yes': 1, 'No': 0})

# Identify categorical and numeric features
categorical_features = X.select_dtypes(include='object').columns.tolist()
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ]
)

# Base models for stacking
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )),
    ('xgb', xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ))
]

# Stacking classifier with meta-model
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    n_jobs=-1
)

# Build the final pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_model)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1)

# Print results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Cross-Validated Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix - Treatment Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()