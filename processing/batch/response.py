import os

from openai import OpenAI
from tqdm import tqdm

from misc import BATCH_REQUESTS_PATH, BATCH_RESPONSES_PATH

client = OpenAI()


def get_file_content(file_id) -> bytes:
    return client.files.content(file_id).content


def download_files() -> None:
    files = client.files.list()

    for file in tqdm(files.data, desc="Downloading batch responses"):
        if file.status == 'processed' and file.purpose == 'batch_output':
            batch_response_file = os.path.join(BATCH_RESPONSES_PATH, f"{file.id}.jsonl")

            if not os.path.exists(batch_response_file):
                with open(batch_response_file, "wb") as f:
                    f.write(get_file_content(file.id))


if __name__ == "__main__":
    requests = len(os.listdir(BATCH_REQUESTS_PATH))
    download_files()
