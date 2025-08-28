import joblib
from pathlib import Path


labels = {0: "ham", 1: "spam"}

def test_model_exists():
    """Check if trained model file exists."""
    model_path = Path("spam_model.joblib")
    assert model_path.exists(), "Trained model file not found."

def test_prediction_spam():
    """Check if spammy message is predicted as spam."""
    model = joblib.load("spam_model.joblib")
    sample_msg = ["Congratulations! You have hunted internship, click link now!"]
    pred_num = model.predict(sample_msg)[0]
    pred = labels[pred_num]
    assert pred == "spam", f"Expected spam but got {pred}"

def test_prediction_ham():
    """Check if normal message is predicted as ham."""
    model = joblib.load("spam_model.joblib")
    sample_msg = ["Hey, are we still meeting at 5 pm today near ELTE?"]
    pred_num = model.predict(sample_msg)[0]
    pred = labels[pred_num]
    assert pred == "ham", f"Expected ham but got {pred}"
