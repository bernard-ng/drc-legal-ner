import argparse
import os
import uuid

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to conditionally run functions.")
    parser.add_argument("--space", action="store_true", help="Clean spacing in files")
    parser.add_argument("--label-studio", action="store_true", help="Convert to Label Studio format")
    args = parser.parse_args()

    if args.space:
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
