import os
from tqdm import tqdm
from openai import OpenAI
client = OpenAI()


def get_content(file_id) -> bytes:
    return client.files.content(file_id).content


def download_files() -> None:
    files = client.files.list()
    for file in tqdm(files.data, desc="Downloading Files"):
        if file.status == 'processed' and file.purpose == 'batch_output':
            content = get_content(file.id)
            with open(f"dataset/responses/{file.id}.jsonl", "wb") as f:
                f.write(content)


if __name__ == "__main__":
    requests = len(os.listdir("dataset/requests"))
    responses = len(os.listdir("dataset/responses"))

    print(f"Preparing to download {responses} out of {requests} files")
    download_files()
