from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
    ClassificationMetrics
)
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train_df = pd.DataFrame(data.train_features)
    y_train_df = pd.DataFrame(data.train_target)
    X_test_df = pd.DataFrame(data.test_features)

    X_train_df['Extracurricular Activities'] = X_train_df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    X_test_df['Extracurricular Activities'] = X_test_df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    X_train_np = X_train_df.to_numpy().astype(float)
    y_train_np = y_train_df.to_numpy().astype(float).flatten()
    X_test_np = X_test_df.to_numpy().astype(float)

    X_mean = np.mean(X_train_np, axis=0)
    X_std = np.std(X_train_np, axis=0)

    X_std[X_std == 0] = 1.0

    X_train_scaled = (X_train_np - X_mean) / X_std
    X_test_scaled = (X_test_np - X_mean) / X_std

    weights = np.zeros(X_train_np.shape[1])
    bias = 0
    loss_history = []
    for i in range(data.epochs):
        y_predict = X_train_scaled @ weights + bias
        
        dweights = 2 * X_train_scaled.T @ (y_predict - y_train_np) / y_train_np.size
        dbias = 2 * (y_predict - y_train_np).sum() / y_train_np.size

        weights = weights - data.learning_rate * dweights
        bias = bias - data.learning_rate * dbias

        y_predict_upd = X_train_scaled @ weights + bias
        mse = ((y_predict_upd - y_train_np) ** 2).sum() / y_train_np.size
        loss_history.append(mse)

    test_predictions = X_test_scaled @ weights + bias
    return RegressionResult(weights=weights,
                            bias=bias,
                            loss_history=loss_history,
                            predictions=test_predictions)

    
def sigmoid(z):
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train_df = pd.DataFrame(data.train_features)
    y_train_df = pd.DataFrame(data.train_target)
    X_test_df = pd.DataFrame(data.test_features)

    X_train_np = X_train_df.to_numpy().astype(float)
    y_train_np = y_train_df.to_numpy().astype(float).flatten()
    X_test_np = X_test_df.to_numpy().astype(float)

    X_mean = np.mean(X_train_np, axis=0)
    X_std = np.std(X_train_np, axis=0)
    
    X_std[X_std == 0] = 1.0
    
    X_train_scaled = (X_train_np - X_mean) / X_std
    X_test_scaled = (X_test_np - X_mean) / X_std

    weights = np.zeros(X_train_np.shape[1])
    bias = 0
    loss_history = []

    for i in range(data.epochs):
        z = X_train_scaled @ weights + bias
        y_predict = sigmoid(z)
        dweights = 1 / y_train_np.size * X_train_scaled.T @ (y_predict - y_train_np)
        dbias = 1 / y_train_np.size * (y_predict - y_train_np).sum()
        weights = weights - data.learning_rate * dweights
        bias = bias - data.learning_rate * dbias

        z_updated = X_train_scaled @ weights + bias
        y_predict_updated = sigmoid(z_updated)
        y_predict_clipped = np.clip(y_predict_updated, 1e-15, 1 - 1e-15)
        logloss = -((y_train_np * np.log(y_predict_clipped) + 
             (1 - y_train_np) * np.log(1 - y_predict_clipped)).sum() / y_train_np.size)
        loss_history.append(logloss)

    probabilities = sigmoid(X_test_scaled @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(weights=weights,
                                bias=bias,
                                loss_history=loss_history,
                                probabilities=probabilities,
                                predictions=predictions)



def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X_train_df = pd.DataFrame(data.train_features)
    y_train_df = pd.DataFrame(data.train_target)
    X_test_df = pd.DataFrame(data.test_features)
    y_test_df = pd.DataFrame(data.test_target)
    
    X_train_np = X_train_df.to_numpy().astype(float)
    y_train_np = y_train_df.to_numpy().astype(float).flatten()
    X_test_np = X_test_df.to_numpy().astype(float)
    y_test_np = y_test_df.to_numpy().astype(float).flatten()
    
    X_mean = np.mean(X_train_np, axis=0)
    X_std = np.std(X_train_np, axis=0)
        
    X_std[X_std == 0] = 1.0
        
    X_train_scaled = (X_train_np - X_mean) / X_std
    X_test_scaled = (X_test_np - X_mean) / X_std

    logistic_regression = train_logistic_classifier(data)
    logistic_regression_metrics = ClassificationMetrics(
        roc_auc=roc_auc_score(y_test_np, logistic_regression.probabilities),
        accuracy=accuracy_score(y_test_np, logistic_regression.predictions),
        precision=precision_score(y_test_np, logistic_regression.predictions, zero_division=0),
        recall=recall_score(y_test_np, logistic_regression.predictions, zero_division=0),
        f1=f1_score(y_test_np, logistic_regression.predictions)
    )

    decision_tree = DecisionTreeClassifier(random_state=42, max_depth=5)
    decision_tree.fit(X_train_scaled, y_train_np)
    decision_tree_predictions = decision_tree.predict(X_test_scaled)
    decision_tree_probabilities = decision_tree.predict_proba(X_test_scaled)[:, 1]
    decision_tree_metrics = ClassificationMetrics(
        roc_auc=roc_auc_score(y_test_np, decision_tree_probabilities),
        accuracy=accuracy_score(y_test_np, decision_tree_predictions),
        precision=precision_score(y_test_np, decision_tree_predictions, zero_division=0),
        recall=recall_score(y_test_np, decision_tree_predictions, zero_division=0),
        f1=f1_score(y_test_np, decision_tree_predictions)
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train_np)
    knn_predictions = knn.predict(X_test_scaled)
    knn_probabilities = knn.predict_proba(X_test_scaled)[:, 1]
    knn_metrics = ClassificationMetrics(
        roc_auc=roc_auc_score(y_test_np, knn_probabilities),
        accuracy=accuracy_score(y_test_np, knn_predictions),
        precision=precision_score(y_test_np, knn_predictions, zero_division=0),
        recall=recall_score(y_test_np, knn_predictions, zero_division=0),
        f1=f1_score(y_test_np, knn_predictions)
    )

    return BaselineComparisonResult(
        logistic_regression=logistic_regression_metrics,
        decision_tree=decision_tree_metrics,
        knn=knn_metrics
    )
