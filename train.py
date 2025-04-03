import random
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example

from misc import TRAINING_EPOCHS, MODEL_NAME, load_json_dataset

# Initialize blank French NLP model
nlp = spacy.blank("fr")

# Ensure NER component is added
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Dataset and model paths
DATASET = load_json_dataset('llm-annotated.json')
random.shuffle(DATASET)

SPLIT_INDEX = int(0.9 * len(DATASET))
TRAIN_SET = DATASET[:SPLIT_INDEX]
EVAL_SET = DATASET[SPLIT_INDEX:]

print(f"Training set size: {len(TRAIN_SET)}")
print(f"Evaluation set size: {len(EVAL_SET)}")


def training():
    for _, annotations in TRAIN_SET:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])  # ent[2] is the entity label

    # Prepare training examples
    db = DocBin()
    for text, annotations in TRAIN_SET:
        try:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label)
                if span:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        except Exception as e:
            continue
    print(f"Used {len(db)} out of {len(TRAIN_SET)} examples")

    # Initialize the model
    optimizer = nlp.initialize()

    # Training loop
    for epoch in range(TRAINING_EPOCHS):
        losses = {}
        random.shuffle(TRAIN_SET)  # Shuffle at each epoch

        for text, annotations in TRAIN_SET:
            try:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.3, losses=losses)
            except Exception:
                continue

        print(f"Epoch {epoch+1} Loss: {losses}")

    nlp.to_disk(MODEL_NAME)
    print(f"Model saved to {MODEL_NAME}")


def evaluation():
    nlp = spacy.load(MODEL_NAME)
    examples = []

    for text, annotations in EVAL_SET:
        try:
            doc = nlp(text)  # Process the text with the pipeline
            example = Example.from_dict(doc, annotations)  # Create an Example
            examples.append(example)
        except Exception:
            continue
    print(f"Used {len(examples)} out of {len(EVAL_SET)} examples")

    # Evaluate the model
    scores = nlp.evaluate(examples)

    print("\n🔍 Evaluation Metrics:")
    print(f"Precision: {scores['ents_p']:.3f}")
    print(f"Recall: {scores['ents_r']:.3f}")
    print(f"F1-score: {scores['ents_f']:.3f}")

    if "ents_per_type" in scores:
        print("\n📌 Per-Entity Scores:")
        for entity, metrics in scores["ents_per_type"].items():
            print(f" - {entity}: Precision={metrics['p']:.3f}, Recall={metrics['r']:.3f}, F1={metrics['f']:.3f}")


if __name__ == "__main__":
    training()
    evaluation()
