import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("training_data.csv")
y = df["match_winner"]
X = df.drop(columns=["match_winner"])

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Feature selection
selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
selector_model.fit(X_scaled, y)
selector = SelectFromModel(selector_model, threshold="median")
X_selected = selector.transform(X_scaled)

selected_features = list(X.columns[selector.get_support()])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# Tune LR
param_grid = {
    "C": [0.1, 1, 10],
    "solver": ["liblinear"],
    "max_iter": [100, 300]
}
lr = GridSearchCV(LogisticRegression(class_weight="balanced"), param_grid, cv=5)
lr.fit(X_train, y_train)
best_lr = lr.best_estimator_

# Other models
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Stacking
stack = StackingClassifier(
    estimators=[("lr", best_lr), ("rf", rf), ("xgb", xgb)],
    final_estimator=LogisticRegression(),
    passthrough=False
)
stack.fit(X_train, y_train)

# Evaluate
y_pred = stack.predict(X_test)
y_prob = stack.predict_proba(X_test)[:, 1]

print(f"✅ Ensemble Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(stack, "ensemble_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "feature_selector.pkl")
joblib.dump(selected_features, "selector_features.pkl")
print("💾 Saved: model, scaler, selector, and selected_features")
