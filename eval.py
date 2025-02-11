import os
import random
import spacy
from spacy.training.example import Example
from tqdm import tqdm
from processing.utils import load_training_data

# Paths
MODEL = './models/leganews-20250211062524'
DATA = load_training_data('./dataset/train.json')

# Ensure the model exists
if not os.path.exists(MODEL):
    raise FileNotFoundError(f"Model not found at {MODEL}")

random.shuffle(DATA)

# Train-test split
split_index = int(0.8 * len(DATA))
EVAL_SET = DATA[split_index:]
print(f"Evaluation set size: {len(EVAL_SET)}")


def evaluation():
    """Evaluate the trained NER model on a test set"""
    nlp = spacy.load(MODEL)
    examples = []
    skipped = 0

    for text, annotations in tqdm(EVAL_SET, desc="Evaluating"):
        try:
            doc = nlp(text)  # Process the text with the pipeline
            example = Example.from_dict(doc, annotations)  # Create an Example
            examples.append(example)
        except Exception as e:
            print(f"Skipping example due to error: {e}")
            skipped += 1

    print(f"Used {len(examples)} out of {len(EVAL_SET)} examples (Skipped: {skipped})")

    # Compute scores
    scores = nlp.evaluate(examples)

    # Print overall metrics
    print("\n🔍 Evaluation Metrics:")
    print(f"Precision: {scores['ents_p']:.3f}")
    print(f"Recall: {scores['ents_r']:.3f}")
    print(f"F1-score: {scores['ents_f']:.3f}")

    # Print detailed entity-wise scores
    if "ents_per_type" in scores:
        print("\n📌 Per-Entity Scores:")
        for entity, metrics in scores["ents_per_type"].items():
            print(f" - {entity}: Precision={metrics['p']:.3f}, Recall={metrics['r']:.3f}, F1={metrics['f']:.3f}")


if __name__ == "__main__":
    evaluation()
