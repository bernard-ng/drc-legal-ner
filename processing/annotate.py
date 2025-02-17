import json
import os
from openai import OpenAI
from tqdm import tqdm
from misc import save_json_dataset, load_csv_dataset, DATA_DIR, load_prompt, OPENAI_MODEL, BATCH_RESPONSES_PATH
from processing.schema import create_named_entities, LegalReference
import argparse

parser = argparse.ArgumentParser(description="Dateset Annotation")
parser.add_argument('--method', type=str, help="annotation method", default='async')
args = parser.parse_args()


client = OpenAI()
prompt = load_prompt()


def annotate(title: str) -> LegalReference:
    """Create a legal reference from the title"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": title},
        ],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content

    return LegalReference.model_validate_json(content)


def annotate_sync() -> list:
    """Annotate dataset synchronously by calling OpenAI API for each row"""

    dataset = load_csv_dataset('data.csv', limit=5)
    annotations = []

    for row in tqdm(dataset, desc="annotating synchronously"):
        try:
            reference = annotate(row['title']) # call to OpenAI API
            annotations.append(create_named_entities(reference))
        except Exception:
            continue

    return annotations


def annotate_async() -> list:
    """Annotate dataset asynchronously by reading from batch responses"""

    annotations = []

    for file in tqdm(os.listdir(BATCH_RESPONSES_PATH), desc="annotating from batch responses"):
        if file.endswith('.jsonl'):
            with open(os.path.join(BATCH_RESPONSES_PATH, file), 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = data['response']['body']['choices'][0]['message']['content']
                        reference = LegalReference.model_validate_json(content)
                        annotations.append(create_named_entities(reference))
                    except Exception:
                        continue

    return annotations


if __name__ == "__main__":
    save_json_dataset(
        data=annotate_sync() if args.method == 'sync' else annotate_async(),
        path='train.json'
    )
