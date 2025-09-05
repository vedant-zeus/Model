import os
import joblib
import pytesseract
from PIL import Image
from LEGAL_JARGONS import LEGAL_JARGONS   # dictionary: jargon → meaning

# -------- Paths --------
MODEL_PATH = os.path.join("..", "models", "saved_model.pkl")
IMAGE_PATH = os.path.join("..", "test_data", "legal_doc.png")  # your input image
OUTPUT_FILE = os.path.join("..", "results", "image_test_results.txt")

# If Tesseract is not in PATH, set it manually (Windows users usually need this):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    """Extract text from a PNG legal document using OCR"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def extract_jargons_from_text(text):
    """Check for legal jargons in extracted text"""
    jargons_found = set()
    for jargon, meaning in LEGAL_JARGONS.items():
        if jargon.lower() in text.lower():
            jargons_found.add(jargon)
    return jargons_found

def test_image():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Please train it first.")
        return

    # Load model (not heavily used here, but kept for consistency)
    vectorizer, clf = joblib.load(MODEL_PATH)

    # Extract text from image
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found at {IMAGE_PATH}")
        return

    text = extract_text_from_image(IMAGE_PATH)

    # Extract jargons
    jargons_found = extract_jargons_from_text(text)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("📌 Image Test Results - Legal Jargons Found:\n\n")
        if not jargons_found:
            out.write("⚠️ No legal jargons found.\n")
            print("⚠️ No legal jargons found.")
        else:
            for jargon in sorted(jargons_found):
                out.write(f"{jargon} → {LEGAL_JARGONS[jargon]}\n")
                print(f"{jargon} → {LEGAL_JARGONS[jargon]}")

    print(f"\n✅ Image test results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    test_image()
