import os
import joblib
from LEGAL_JARGONS import LEGAL_JARGONS   # dictionary: jargon → meaning

# Paths
MODEL_PATH = os.path.join("..", "models", "saved_model.pkl")
TEST_DATA_FOLDER = os.path.join("..", "test_data")   # folder containing test .txt files
OUTPUT_FILE = os.path.join("..", "results", "test_results.txt")

def extract_jargons_from_file(file_path):
    """Extract jargons from a single file"""
    jargons_found = set()
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines:
        for jargon, meaning in LEGAL_JARGONS.items():
            if jargon.lower() in line.lower():
                jargons_found.add(jargon)
    return jargons_found

def test_dataset():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Please train it first.")
        return

    # Load model
    vectorizer, clf = joblib.load(MODEL_PATH)

    # Prepare output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("📌 Test Results - Legal Jargons Found:\n\n")

        # Loop through all test files
        for file_name in os.listdir(TEST_DATA_FOLDER):
            if file_name.endswith(".txt"):
                file_path = os.path.join(TEST_DATA_FOLDER, file_name)
                jargons_found = extract_jargons_from_file(file_path)

                out.write(f"File: {file_name}\n")
                print(f"\n📂 File: {file_name}")

                if not jargons_found:
                    out.write("   ⚠️ No legal jargons found.\n\n")
                    print("   ⚠️ No legal jargons found.")
                else:
                    for jargon in sorted(jargons_found):
                        out.write(f"   {jargon} → {LEGAL_JARGONS[jargon]}\n")
                        print(f"   {jargon} → {LEGAL_JARGONS[jargon]}")
                    out.write("\n")

    print(f"\n✅ Test results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    test_dataset()
