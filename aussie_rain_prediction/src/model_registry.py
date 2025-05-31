from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(solver="liblinear"),
    "random_forest": RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        n_estimators=1000,
        max_features=7,
        max_depth=30,
        class_weight={"No": 1, "Yes": 1.5},
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=7, max_leaf_nodes=128, random_state=42
    ),
}
