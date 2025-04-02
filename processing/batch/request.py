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


def cancel_batches():
    try:
        limit, after, cancelled_count = 100, None, 0
        while True:
            batches = client.batches.list(limit=limit, after=after)
            if not batches.data:
                print("No active batch jobs found.")
                break

            for batch in batches.data:
                batch_id = batch.id
                status = batch.status

                if status in ["in_progress", "queued"]:  # Cancel only running/queued jobs
                    print(f"Cancelling batch job: {batch_id} (Status: {status})")
                    client.batches.cancel(batch_id)
                    cancelled_count += 1
                else:
                    print(f"Skipping batch {batch_id} (Status: {status})")

            if batches.has_next_page():
                next_page_info = batches.next_page_info()
                if next_page_info:
                    after = next_page_info.params.get("after")
                else:
                    break
            else:
                break

        print(f"Total {cancelled_count} running or queued batch jobs have been cancelled.")

    except Exception as e:
        print(f"Error: {e}")


def create_batches() -> None:
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


def retry_batches():
    with open(CREATED_BATCHES_LOGS_PATH, "r") as f:
        ids = f.readlines()[1:]

    failed_batches_ids = []
    retried_batches_ids = []

    for batch_id in tqdm(ids, desc="Retrying failed batches"):
        batch = client.batches.retrieve(batch_id.strip())

        if batch.status == "failed" and batch.input_file_id:
            retried_batch = client.batches.create(
                input_file_id=batch.input_file_id.strip(),
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            if retried_batch:
                failed_batches_ids.append(batch.id)
                retried_batches_ids.append(retried_batch.id)

    if failed_batches_ids:
        with open(CREATED_BATCHES_LOGS_PATH, "r") as f:
            lines = f.readlines()

        with open(CREATED_BATCHES_LOGS_PATH, "w") as f:
            for line in lines:
                batch_id = line.strip()
                if batch_id in failed_batches_ids:
                    f.write(f"{retried_batches_ids[failed_batches_ids.index(batch_id)]}\n")
                else:
                    f.write(line)


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
    parser.add_argument("--build", action="store_true", help="Create JSONL requests")
    parser.add_argument("--upload", action="store_true", help="Upload JSONL requests")
    parser.add_argument("--create", action="store_true", help="Create batch jobs")
    parser.add_argument('--cancel', action='store_true', help='Cancel all running batch jobs')
    parser.add_argument('--retry', action='store_true', help='Retry all failed batch jobs')

    args = parser.parse_args()

    if args.build:
        build_jsonl_requests()
    if args.upload:
        upload_jsonl_requests()
    if args.create:
        create_batches()
    if args.cancel:
        cancel_batches()
    if args.retry:
        retry_batches()
