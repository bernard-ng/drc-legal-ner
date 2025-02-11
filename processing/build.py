import json
import os

from utils import load_dataset, extract_reference, reference_to_named_entities, save_training_data, LegalReference
import argparse

parser = argparse.ArgumentParser(description="Build training data")
parser.add_argument('--type', type=str, help="building method", default='async')
args = parser.parse_args()

building_type = 'async' if args.type is None else args.type


def build_training_data_sync() -> list:
    training_data = []
    data = load_dataset('../dataset/data.csv', 5)

    for row in data:
        try:
            reference = extract_reference(row['title'])
            training_data.append(reference_to_named_entities(reference))
            print("\n")
        except Exception as e:
            print(f">> Error: {e}")
            continue

    return training_data


def build_training_data_async() -> list:
    training_data = []
    files = os.listdir('batch/dataset/responses')

    for file in files:
        if file.endswith('.jsonl'):
            with open(f'batch/dataset/responses/{file}', 'r') as f:
                for line in f:
                    try:
                        response = json.loads(line)
                        content = response['response']['body']['choices'][0]['message']['content']
                        reference = LegalReference.model_validate_json(content)
                        training_data.append(reference_to_named_entities(reference))
                    except Exception as e:
                        print(f">> Error: {e}")
                        continue
                print('\n')

    return training_data


if __name__ == "__main__":
    save_training_data(
        data=build_training_data_sync() if building_type == 'sync' else build_training_data_async(),
        path='../dataset/train.json'
    )
