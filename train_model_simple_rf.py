
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and prepare data
df = pd.read_csv("training_data.csv")
df = df.drop(columns=["team1", "team2", "confidence"], errors="ignore")
X = df.drop(columns=["match_winner"])
y = df["match_winner"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ RandomForest Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "ensemble_model.pkl")
print("💾 Saved RandomForest model as ensemble_model.pkl")
