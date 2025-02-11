import os
import spacy

MODEL_PATH = "./models/leganews-20250211062524"

# Ensure the model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

nlp = spacy.load(MODEL_PATH)

print("\n📌 Named Entity Recognition (NER) - Legal Titles")
print("🔹 Type a legal document title and get entity predictions.")
print("🔹 Press Enter without typing anything to exit.\n")

while True:
    try:
        text = input("📜 Enter title: ").strip()
        if not text:
            print("\n👋 Exiting NER tool. Goodbye!\n")
            break

        doc = nlp(text)

        if doc.ents:
            print("\n🔍 Recognized Entities:")
            for ent in doc.ents:
                print(f" - **{ent.text}** (Label: {ent.label_})")
        else:
            print("\n⚠️ No entities detected.")

        print("\n" + "=" * 40 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
