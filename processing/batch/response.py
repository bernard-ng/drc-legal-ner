import os
from tqdm import tqdm
from openai import OpenAI

from misc import BATCH_REQUESTS_PATH, BATCH_RESPONSES_PATH

client = OpenAI()


def get_file_content(file_id) -> bytes:
    return client.files.content(file_id).content


def download_files() -> None:
    files = client.files.list()

    for file in tqdm(files.data, desc="Downloading batch responses"):
        if file.status == 'processed' and file.purpose == 'batch_output':
            batch_response_file = os.path.join(BATCH_RESPONSES_PATH, f"{file.id}.jsonl")

            with open(batch_response_file, "wb") as f:
                f.write(get_file_content(file.id))


if __name__ == "__main__":
    requests = len(os.listdir(BATCH_REQUESTS_PATH))
    responses = len(os.listdir(BATCH_RESPONSES_PATH))

    print(f"Preparing to download {responses} out of {requests} files")
    download_files()
