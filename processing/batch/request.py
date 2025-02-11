import json
from processing.utils import load_dataset
from tqdm import tqdm
import os
from openai import OpenAI

client = OpenAI()
BATCH_SIZE = 10

data = load_dataset('../../dataset/data.csv')
with open('../prompt.txt', 'r') as f:
    prompt = f.read()


def build_request(identifier: int, title: str) -> str:
    request = {
        "custom_id": f"{identifier}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": title}
            ],
            "response_format": {"type": "json_object"}
        }
    }

    return json.dumps(request, ensure_ascii=False, separators=(',', ':'))


def save_requests(r: list, path: str) -> None:
    for i in range(0, len(r), BATCH_SIZE):
        with open(f"{path[:-6]}-{i}.jsonl", "w", encoding="utf-8") as f:
            f.write('\n'.join(r[i:i+BATCH_SIZE]))


def build_requests_jsonl(path: str) -> None:
    requests = []
    for i, row in enumerate(data):
        requests.append(build_request(i, row['title']))

    save_requests(requests, path)


def create_batches() -> None:
    with open("dataset/logs/last-uploaded-files.csv", "r") as f:
        ids = f.readlines()[1:]

    responses = []
    for file in tqdm(ids, desc="Creating Batches"):
        file_id = file.strip()
        response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        responses.append(response)

    print(">> Batch Created !")
    with open("dataset/logs/last-created-batches.csv", "w") as f:
        f.write("batch_id\n")
        for response in tqdm(responses, desc="Saving Batch IDs"):
            f.write(f"{response.id}\n")


def delete_last_uploaded_files() -> None:
    with open("dataset/logs/last-uploaded-files.csv", "r") as f:
        ids = f.readlines()[1:]
        for file_id in tqdm(ids, desc="Deleting Files"):
            client.files.delete(file_id.strip())


def upload_files() -> None:
    responses = []
    for file in tqdm(os.listdir("dataset/requests"), desc="Uploading Files"):
        if file.endswith(".jsonl"):
            response = client.files.create(file=open(f"dataset/requests/{file}", "rb"), purpose="batch")
            responses.append(response)

    with open("dataset/logs/last-uploaded-files.csv", "w") as f:
        f.write("file_id\n")
        for response in tqdm(responses, desc="Saving File IDs"):
            f.write(f"{response.id}\n")


if __name__ == '__main__':
    build_requests_jsonl('/dataset/requests/requests.jsonl')
    upload_files()
    create_batches()
