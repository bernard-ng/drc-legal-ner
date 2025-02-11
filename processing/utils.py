import json
import csv
from pydantic import BaseModel
from openai import OpenAI

MODEL = "gpt-3.5-turbo" # "mistral"
PROVIDER = 'https://api.openai.com/v1' # 'http://localhost:11434/v1'


class LegalReference(BaseModel):
  title: str
  reference: str
  type: str
  date: str


def extract_reference(title: str) -> LegalReference:
    print(f">> Extracting reference from title: {title}")

    prompt = load_prompt()
    client = OpenAI(base_url=PROVIDER)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": title},
        ],
        response_format=LegalReference.model_json_schema() if 'localhost' in PROVIDER else {"type": "json_object"}
    )

    content = response.choices[0].message.content
    return LegalReference.model_validate_json(content)


def reference_to_named_entities(reference: LegalReference) -> tuple:
    print(f">> Converting Named Entities == {reference.reference}, {reference.type}, {reference.date}")
    title = reference.title
    named_entities = [
        (title.index(reference.reference), title.index(reference.reference) + len(reference.reference), "REFERENCE"),
        (title.index(reference.type), title.index(reference.type) + len(reference.type), "TYPE"),
        (title.index(reference.date), title.index(reference.date) + len(reference.date), "DATE")
    ]

    return title, {"entities": named_entities}


def save_training_data(data: list, path: str) -> None:
    print(f">> Saving training data to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))


def load_prompt() -> str:
    with open('prompt.txt', 'r') as f:
        return f.read()


def load_dataset(path: str, limit: int = None) -> list:
    print(f">> Loading dataset from {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            if limit and len(data) >= limit:
                break

    return data


def load_training_data(path: str) -> list:
    print(f">> Loading training data from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
