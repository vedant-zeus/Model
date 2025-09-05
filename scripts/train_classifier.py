import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from LEGAL_JARGONS import LEGAL_JARGONS   # dictionary: jargon → meaning

# Paths
RAW_DATA_FOLDER = os.path.join("..", "data")   # folder containing training .txt files
MODEL_PATH = os.path.join("..", "models", "saved_model.pkl")
OUTPUT_FOLDER = os.path.join("..", "results")
RESULT_FILE = os.path.join(OUTPUT_FOLDER, "trained_jargons.txt")

def load_and_extract_jargons(folder_path):
    """Load all .txt files and extract only jargons with their meanings"""
    jargon_sentences = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()

            for line in lines:
                for jargon, meaning in LEGAL_JARGONS.items():
                    if jargon.lower() in line.lower():
                        # Save only jargon word (not full line)
                        jargon_sentences.append(jargon)
                        labels.append(1)  # label for "legal jargon"
    
    return jargon_sentences, labels

def train_classifier():
    # Extract only jargons
    texts, labels = load_and_extract_jargons(RAW_DATA_FOLDER)

    if not texts:
        print("⚠️ No jargons found in dataset. Please check LEGAL_JARGONS dictionary and data files.")
        return

    # Vectorize
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Train classifier
    clf = MultinomialNB()
    clf.fit(X, labels)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((vectorizer, clf), MODEL_PATH)

    # Save results into a file
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("📌 Trained Jargons with Meanings:\n\n")
        for jargon in sorted(set(texts)):
            f.write(f"{jargon} → {LEGAL_JARGONS[jargon]}\n")

    print(f"✅ Model trained only on legal jargons and saved at {MODEL_PATH}")
    print(f"📂 Jargon results saved in {RESULT_FILE}")

if __name__ == "__main__":
    train_classifier()
