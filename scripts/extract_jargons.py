import os
from LEGAL_JARGONS import LEGAL_JARGONS

# Paths
RAW_DATA_PATH = os.path.join("..", "data","1950_Ashutosh_Lahiry_vs_The_State_Of_Delhi_And_Anr_on_19_May_1950_1.segmented.txt")
OUTPUT_PATH = os.path.join("..", "outputs", "jargon_matches.txt")

def extract_jargons_from_text(text, jargon_list):
    """Extracts legal jargons from given text."""
    found = []
    for jargon in jargon_list:
        if jargon.lower() in text.lower():
            found.append(jargon)
    return list(set(found))

def main():
    # Load dataset
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract jargons
    jargons_found = extract_jargons_from_text(text, LEGAL_JARGONS)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for jargon in jargons_found:
            f.write(jargon + "\n")

    print(f"✅ Extracted {len(jargons_found)} jargons. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
