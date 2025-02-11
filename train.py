from datetime import datetime
import random
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from tqdm import tqdm
from processing.utils import load_training_data


# Initialize blank French NLP model
nlp = spacy.blank("fr")

# Ensure NER component is added
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Dataset and model paths
EPOCHS = 5
MODEL = f"./models/leganews-{datetime.now().strftime('%Y%m%d%H%M%S')}"
DATA = load_training_data('./dataset/train.json')
random.shuffle(DATA)

split_index = int(0.8 * len(DATA))
TRAIN_SET = DATA[:split_index]
print(f"Training set size: {len(TRAIN_SET)}")


def training():
    """Train a named entity recognition (NER) model with spaCy"""
    # Add entity labels from training data
    for _, annotations in TRAIN_SET:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])  # ent[2] is the entity label

    # Prepare training examples
    db = DocBin()
    used = 0
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
            used += 1
        except Exception as e:
            print(f"Skipping example due to error: {e}")
            continue

    print(f"Used {used} out of {len(TRAIN_SET)} examples")

    # Initialize the model
    optimizer = nlp.initialize()

    # Training loop
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        losses = {}
        random.shuffle(TRAIN_SET)  # Shuffle at each epoch
        for text, annotations in TRAIN_SET:
            try:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.3, losses=losses)
            except Exception as e:
                print(f"Skipping training example due to error: {e}")
                continue

        print(f"Epoch {epoch+1} Loss: {losses}")

    # Save the trained model
    nlp.to_disk(MODEL)
    print(f"Model saved to {MODEL}")


if __name__ == "__main__":
    training()
