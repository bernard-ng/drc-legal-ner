import argparse
import json
import os

from openai import OpenAI
from tqdm import tqdm

from misc import save_json_dataset, load_csv_dataset, load_prompt, OPENAI_MODEL, BATCH_RESPONSES_PATH
from misc.model import LegalReference


def annotate(client: OpenAI, prompt: str, title: str) -> LegalReference:
    """Create a legal reference from the title"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": title},
        ],
        response_format={"type": "json_object"} if OPENAI_MODEL == 'gpt-3.5-turbo' else LegalReference
    )

    if OPENAI_MODEL == 'gpt-3.5-turbo':
        content = response.choices[0].message.content
        return LegalReference.model_validate_json(content)
    else:
        return response.choices[0].message.parsed


def annotate_sync() -> list:
    """Annotate dataset synchronously by calling OpenAI API for each row"""

    client = OpenAI()
    prompt = load_prompt()
    dataset = load_csv_dataset('data.csv', limit=5)
    annotations = []

    for row in tqdm(dataset, desc="Annotating synchronously"):
        try:
            reference = annotate(client, prompt, row['title'])  # Call to OpenAI API
            annotations.append(reference.to_named_entities())
        except Exception as e:
            print(f'Unable to annotate: {e}')
            continue

    return annotations


def annotate_async() -> list:
    """Annotate dataset asynchronously by reading from batch responses"""

    annotations = []

    for file in tqdm(os.listdir(BATCH_RESPONSES_PATH), desc="Annotating from batch responses"):
        if file.endswith('.jsonl'):
            with open(os.path.join(BATCH_RESPONSES_PATH, file), 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = data['response']['body']['choices'][0]['message']['content']
                        reference = LegalReference.model_validate_json(content)
                        annotations.append(reference.to_named_entities())
                    except Exception as e:
                        print(f'Unable to parse response: {e}')
                        continue

    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Annotation")
    parser.add_argument('--method', type=str, help="Annotation method: 'sync' or 'async'", default='async')
    args = parser.parse_args()

    save_json_dataset(
        data=annotate_sync() if args.method == 'sync' else annotate_async(),
        path='llm-annotated.json'
    )
