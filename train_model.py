from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("datasets/Training_augmented.csv") 
X = df.drop(columns=['prognosis'])
y = df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

xgb_disease_model= xgb.XGBClassifier(eval_metric='mlogloss')
xgb_disease_model.fit(X_train, y_train)

y_pred = xgb_disease_model.predict(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

with open("models/rf_disease_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("models/xgb_disease_model.pkl", "wb") as f:
    pickle.dump(xgb_disease_model, f)

with open("models/le.pkl", "wb") as f:
    pickle.dump(le, f)


print("Models trained and saved successfully.")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

