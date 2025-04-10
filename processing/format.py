import argparse
import os
import uuid
import random

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

from misc import clean_spacing, DATA_DIR
from misc import load_json_dataset, save_json_dataset


def convert_label_studio_to_spacy(label_studio_data):
    text = label_studio_data.get("data", {}).get("text", "")
    results = label_studio_data.get("predictions", [{}])[0].get("result", [])

    entities = [
        (res["value"]["start"], res["value"]["end"], label)
        for res in results
        for label in res["value"].get("labels", [])
    ]

    return [text, {"entities": entities}]


def convert_to_label_studio_format(data):
    text, annotations = data
    entities = annotations.get("entities", [])

    converted = {
        "data": {"text": text},
        "predictions": [
            {
                "model_version": "one",
                "score": 1.0,
                "result": [
                    {
                        "id": str(uuid.uuid4())[:8],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "score": 1.0,
                            "text": text[start:end],
                            "labels": [label]
                        }
                    }
                    for start, end, label in entities
                ]
            }
        ]
    }
    return converted


def convert_to_spacy_binary_format(path, data):
    nlp = spacy.blank("fr")
    db = DocBin()

    for text, annotations in tqdm(data):
        try:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
        except Exception:
            continue

    db.to_disk(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to conditionally run functions.")
    parser.add_argument("--normalize-space", action="store_true", help="Clean spacing in files")
    parser.add_argument("--label-studio", action="store_true", help="Convert to Label Studio format")
    parser.add_argument('--spacy-binary', action="store_true", help="Convert to Spacy binary format")
    args = parser.parse_args()

    if args.normalize_space:
        files = ['data.csv', 'pre-annotated.json', 'llm-annotated.json', 'prompt.txt'];
        for file_name in tqdm(files, desc="cleaning files"):
            file_path = os.path.join(DATA_DIR, file_name)
            cleaned_content = clean_spacing(file_name)

            if cleaned_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

    if args.label_studio:
        data = []
        for annotation in load_json_dataset('llm-annotated.json'):
            data.append(convert_to_label_studio_format(annotation))

        save_json_dataset(data, 'pre-annotated.json')

    if args.spacy_binary:
        dataset = load_json_dataset('llm-annotated.json')
        random.shuffle(dataset)

        index = int(0.8 * len(dataset))
        train_set = dataset[:index]
        dev_set = dataset[index:]

        convert_to_spacy_binary_format(os.path.join(DATA_DIR, 'spacy/train.spacy'), train_set)
        convert_to_spacy_binary_format(os.path.join(DATA_DIR, 'spacy/dev.spacy'), dev_set)
