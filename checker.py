import pandas as pd, joblib
from pathlib import Path

FEATURES = ["amount", "hour", "is_foreign", "merchant_risk"]  # match your CSV columns
MODEL_PATH = "models/checker.joblib"

def train(csv="data/sample_transactions.csv"):
    # sklearn is imported here, not at module scope: it costs ~84 MB of RSS and only
    # training needs it up front. joblib.load pulls in what unpickling requires.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    df = pd.read_csv(csv)
    X, y = df[FEATURES], df["label"]                   # label column: 1 = suspicious
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte)))          # precision / recall / F1
    print("ROC-AUC:", roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def flag_transactions(df: pd.DataFrame):
    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(df[FEATURES])[:, 1]               # risk score per row
    df = df.assign(risk_score=scores)
    return df[df.risk_score > 0.5].sort_values("risk_score", ascending=False)

if __name__ == "__main__":
    train()   # run once: python checker.py