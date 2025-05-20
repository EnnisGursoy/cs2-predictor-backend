
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load training data
df = pd.read_csv("training_data.csv")
df = df.drop(columns=["team1", "team2", "confidence"], errors="ignore")
X = df.drop(columns=["match_winner"])
y = df["match_winner"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune XGBoost
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", use_label_encoder=False),
    param_grid=param_grid,
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Show top 10 features
importances = model.feature_importances_
feature_names = X.columns
top_features = sorted(zip(importances, feature_names), reverse=True)[:10]
print("\n🔥 Top 10 Features:")
for score, name in top_features:
    print(f"  {name}: {score:.4f}")

# Save model
joblib.dump(model, "ensemble_model.pkl")
print("💾 Saved XGBoost model as ensemble_model.pkl")
