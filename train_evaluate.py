import joblib
from data.data_processor import DataProcessor
from models.random_forest import RandomForestModel
from models.xgboost import XGBoostModel
from sklearn.metrics import classification_report, roc_auc_score

# Initialize Data Processor
data_processor = DataProcessor("data/creditcard.csv")

# Preprocess data
X, y = data_processor.preprocess_data()
X_train, X_test, y_train, y_test = data_processor.get_train_test_split(X, y)
X_train, y_train = data_processor.apply_smote(X_train, y_train)

# Define models
models = {
    "random_forest": RandomForestModel(n_estimators=100, max_depth=10),
    "xgboost": XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=6)
}

# Train and evaluate
best_model = None
best_roc_auc = 0
for model_name, model_instance in models.items():
    model_instance.build_model()
    model_instance.train(X_train, y_train)
    predictions = model_instance.predict(X_test)
    probabilities = model_instance.predict_proba(X_test)[:, 1]

    # Evaluate
    print(f"Model: {model_name}")
    print(classification_report(y_test, predictions))
    roc_auc = roc_auc_score(y_test, probabilities)
    print(f"ROC-AUC Score: {roc_auc}\n")

    # Save best model
    if roc_auc > best_roc_auc:
        best_model = model_instance
        best_roc_auc = roc_auc
        joblib.dump(model_instance.model, f"app/saved_models/{model_name}_best_model.joblib")

print(f"Best model saved: {best_model}")
