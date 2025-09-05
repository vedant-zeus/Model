import os
import joblib

MODEL_PATH = os.path.join("..", "models", "saved_model.pkl")
RAW_DATA_PATH = os.path.join("..", "data","1950_Ashutosh_Lahiry_vs_The_State_Of_Delhi_And_Anr_on_19_May_1950_1.segmented.txt")

def evaluate():
    vectorizer, clf = joblib.load(MODEL_PATH)

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        text = f.readlines()

    X = vectorizer.transform(text)
    preds = clf.predict(X)

    print(f"✅ Evaluated {len(preds)} samples")

if __name__ == "__main__":
    evaluate()
