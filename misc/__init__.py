import csv
import json
import os
from datetime import datetime
from typing import Optional

# Paths
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
BATCH_DATA_DIR = os.path.join(ROOT_DIR, 'dataset/batch')

BATCH_REQUESTS_PATH = os.path.join(BATCH_DATA_DIR, 'requests')
BATCH_RESPONSES_PATH = os.path.join(BATCH_DATA_DIR, 'responses')
UPLOADED_REQUESTS_LOGS_PATH = os.path.join(BATCH_DATA_DIR, 'logs/uploaded-requests.csv')
CREATED_BATCHES_LOGS_PATH = os.path.join(BATCH_DATA_DIR, 'logs/created-batches.csv')

# LLM
OPENAI_MODEL = "gpt-4o-mini"

# Training
TRAINING_EPOCHS = 5
BATCH_REQUESTS_SIZE = 5
MODEL_NAME = f"./models/leganews-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def clean_spacing(filename: str) -> Optional[str]:
    try:
        with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf8') as f:
            content = f.read()
            return content.translate(str.maketrans({'\00': ' ', ' ': ' '}))
    except Exception as e:
        return None


def load_json_dataset(path: str) -> list:
    print(f">> Loading JSON dataset from {path}")
    with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_dataset(data: list, path: str) -> None:
    print(f">> Saving CSV dataset to {path}")
    with open(os.path.join(DATA_DIR, path), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_csv_dataset(path: str, limit: int = None) -> list:
    print(f">> Loading CSV dataset from {path}")
    data = []
    with open(os.path.join(DATA_DIR, path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            if limit and len(data) >= limit:
                break

    return data


def load_prompt() -> str:
    with open(os.path.join(DATA_DIR, 'prompt.txt'), 'r') as f:
        return f.read()


def save_json_dataset(data: list, path: str) -> None:
    print(f">> Saving JSON dataset to {path}")
    with open(os.path.join(DATA_DIR, path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
