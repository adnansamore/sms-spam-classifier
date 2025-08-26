import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from joblib import dump

def load_dataset(path: str) -> pd.DataFrame:
    def _try_read(sep, names=None):
        try:
            return pd.read_csv(path, sep=sep, encoding="latin-1", names=names)
        except Exception:
            return None

    df = None
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception:
        df = None

    if df is not None:
        #columns: v1(label), v2(text), plus extra unnamed cols
        if {"v1", "v2"}.issubset(df.columns):
            df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        elif df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["label", "text"]
    else:
        df = _try_read("\t", names=["label", "text"])

    if df is None or df.shape[1] < 2:
        raise ValueError("Could not read dataset. Please provide Kaggle spam.csv or UCI SMSSpamCollection.")

    # Basic clean-up
    df = df.dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    # Deduplicate
    df = df.drop_duplicates(subset=["text"])
    return df

def build_pipeline(model="lr"):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        strip_accents="unicode",
        max_df=0.95,
        min_df=2
    )
    if model == "lr":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = MultinomialNB()
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])

def evaluate(y_true, y_pred, title="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("\nConfusion matrix [ [TN FP]\n                   [FN TP] ]")
    print(confusion_matrix(y_true, y_pred))
    return acc, prec, rec, f1

def main():
    parser = argparse.ArgumentParser(description="Train SMS Spam Classifier")
    parser.add_argument("--data_path", required=True, help="Path to spam dataset.")
    parser.add_argument("--model_path", default="spam_model.joblib", help="Where to save the trained model pipeline.")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_dataset(args.data_path)
  
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    X = df["text"].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Two pipelines: Logistic Regression and Naive Bayes
    lr_pipe = build_pipeline("lr")
    nb_pipe = build_pipeline("nb")

    print("Training Logistic Regression pipeline...")
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    _, _, _, lr_f1 = evaluate(y_test, lr_pred, title="Logistic Regression")

    print("\nTraining Naive Bayes pipeline...")
    nb_pipe.fit(X_train, y_train)
    nb_pred = nb_pipe.predict(X_test)
    _, _, _, nb_f1 = evaluate(y_test, nb_pred, title="Multinomial Naive Bayes")

    # Pick best by F1
    best_pipe, best_name, best_pred = (lr_pipe, "Logistic Regression", lr_pred) if lr_f1 >= nb_f1 else (nb_pipe, "Naive Bayes", nb_pred)

    print(f"\nBest model: {best_name}")
    print("\nClassification report (best model):")
    print(classification_report(y_test, best_pred, target_names=["ham", "spam"]))

    # Saving the full pipeline
    dump(best_pipe, args.model_path)
    print(f"\nSaved model pipeline to: {os.path.abspath(args.model_path)}")

    # testing
    samples = [
        "URGENT! You have won a free ticket. Claim now!",
        "Are we meeting at 5 pm near the ELTE campus?",
        "Congratulations, you have been selected for a $1000 gift card. Reply WIN."
    ]
    preds = best_pipe.predict(samples)
    proba = best_pipe.predict_proba(samples)[:, 1]
    print("\nSample predictions:")
    for s, p, pr in zip(samples, preds, proba):
        label = "spam" if p == 1 else "ham"
        print(f"- '{s[:60]}...' -> {label} (spam probability={pr:.3f})")

if __name__ == "__main__":
    main()
