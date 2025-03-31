from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

if __name__ == "__main__":
    response = client.files.list()

    for file in tqdm(response, desc="Deleting Files"):
        try:
            client.files.delete(file.id)
        except Exception as e:
            print(f'Error : {e}')
            continue
