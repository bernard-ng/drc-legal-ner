import argparse
import json
import os

from openai import OpenAI
from openai.lib._pydantic import \
    to_strict_json_schema  # https://community.openai.com/t/structured-outputs-with-batch-processing/911076/10
from tqdm import tqdm

from misc import load_csv_dataset, load_prompt, BATCH_REQUESTS_SIZE, BATCH_DATA_DIR, \
    UPLOADED_REQUESTS_LOGS_PATH, CREATED_BATCHES_LOGS_PATH, BATCH_REQUESTS_PATH, OPENAI_MODEL
from misc.model import LegalReference

client = OpenAI()
data = load_csv_dataset('data.csv')
prompt = load_prompt()


def build_request(identifier: int, title: str) -> str:
    if OPENAI_MODEL == 'gpt-3.5-turbo':
        schema = {"type": "json_object"}
    else:
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": 'LegalReference',
                "schema": to_strict_json_schema(LegalReference),
                "strict": True
            }
        }


    request = {
        "custom_id": f"{identifier}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": title}
            ],
            "response_format": schema
        }
    }

    return json.dumps(request, ensure_ascii=False, separators=(',', ':'))


def build_jsonl_requests() -> None:
    requests = []
    for i, row in enumerate(data):
        requests.append(build_request(i, row['title']))

    for i in tqdm(range(0, len(requests), BATCH_REQUESTS_SIZE), desc="Creating JSONL request files"):
        request_file = os.path.join(BATCH_DATA_DIR, f"requests/request-{i}.jsonl")

        with open(request_file, "w", encoding="utf-8") as f:
            f.write('\n'.join(requests[i:i + BATCH_REQUESTS_SIZE]))


def create_batch_jobs() -> None:
    with open(UPLOADED_REQUESTS_LOGS_PATH, "r") as f:
        ids = f.readlines()[1:]

    responses = []
    for file_id in tqdm(ids, desc="Creating Batches for uploaded requests"):
        response = client.batches.create(
            input_file_id=file_id.strip(),
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        responses.append(response)

    with open(CREATED_BATCHES_LOGS_PATH, "w") as f:
        f.write("batch_id\n")
        for batch in tqdm(responses, desc="Saving Batch IDs"):
            f.write(f"{batch.id}\n")


def upload_jsonl_requests() -> None:
    responses = []

    for file in tqdm(os.listdir(BATCH_REQUESTS_PATH), desc="Uploading JSONL request files"):
        if file.endswith(".jsonl"):
            batch_request_file = open(os.path.join(BATCH_REQUESTS_PATH, file), "rb")
            response = client.files.create(file=batch_request_file, purpose="batch")
            responses.append(response)

    with open(UPLOADED_REQUESTS_LOGS_PATH, "w") as f:
        f.write("file_id\n")
        for file in tqdm(responses, desc="Saving File IDs"):
            f.write(f"{file.id}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to conditionally run functions.")
    parser.add_argument("--build", action="store_true", help="Run build_jsonl_requests()")
    parser.add_argument("--upload", action="store_true", help="Run upload_jsonl_requests()")
    parser.add_argument("--batch", action="store_true", help="Run create_batch_jobs()")

    args = parser.parse_args()

    if args.build:
        build_jsonl_requests()
    if args.upload:
        upload_jsonl_requests()
    if args.batch:
        create_batch_jobs()
