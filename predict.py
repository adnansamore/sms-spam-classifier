import argparse
from joblib import load

def main():
    parser = argparse.ArgumentParser(description="Predict spam/ham for SMS text(s)")
    parser.add_argument("--model_path", default="spam_model.joblib")
    parser.add_argument("texts", nargs="+", help="One or more messages to classify (quote each).")
    args = parser.parse_args()

    pipe = load(args.model_path)
    preds = pipe.predict(args.texts)
    probas = pipe.predict_proba(args.texts)[:, 1]

    for text, p, prob in zip(args.texts, preds, probas):
        label = "spam" if p == 1 else "ham"
        print(f"Text: {text}\n→ Prediction: {label} | spam probability={prob:.3f}\n")

if __name__ == "__main__":
    main()
