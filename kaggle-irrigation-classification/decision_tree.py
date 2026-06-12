import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

print("\n--- MISSING VALUES ---")
print(train.isnull().sum())

num_cols = train.select_dtypes(include=np.number).columns
for col in num_cols:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(test[col].median())

cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(test[col].mode()[0])

combined = pd.concat([train.drop("Irrigation_Need", axis=1), test], axis=0)
le_dict = {}
for col in combined.select_dtypes(include='object').columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    le_dict[col] = le

X = combined.iloc[:len(train), :]
test_processed = combined.iloc[len(train):, :]

target_le = LabelEncoder()
y = target_le.fit_transform(train["Irrigation_Need"])

test_ids = test_processed["id"]
X = X.drop("id", axis=1)
test_processed = test_processed.drop("id", axis=1)

dt_model = DecisionTreeClassifier()  
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(dt_model, X, y, cv=kf, scoring='accuracy')
print("Scores:", scores)
print("Mean Accuracy:", scores.mean())
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model.fit(X_train, y_train)
val_pred = dt_model.predict(X_val)

print("\nValidation Accuracy:", accuracy_score(y_val, val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))
dt_model.fit(X, y)
preds = dt_model.predict(test_processed)
preds_final = target_le.inverse_transform(preds)

submission = pd.DataFrame({
    "id": test_ids,
    "irrigation_need": preds_final
})

submission.to_csv("submission_dt_basic.csv", index=False)

print("\nsubmission_dt_basic.csv created successfully!")