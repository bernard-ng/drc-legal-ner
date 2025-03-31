import uuid

from misc import load_json_dataset, save_json_dataset


def convert_label_studio_to_spacy(label_studio_data):
    text = label_studio_data["data"]["text"]
    results = label_studio_data["predictions"][0]["result"]

    entities = [
        [res["value"]["start"], res["value"]["end"], res["value"]["labels"][0]]
        for res in results
    ]

    return [text, {"entities": entities}]


def convert_to_label_studio_format(data):
    text, annotations = data
    entities = annotations["entities"]

    converted = {
        "data": {
            "text": text
        },
        "predictions": [
            {
                "model_version": "one",
                "score": 1.0,
                "result": [
                    {
                        "id": str(uuid.uuid4())[:8],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "score": 1.0,
                            "text": text[start:end],
                            "labels": [label]
                        }
                    }
                    for start, end, label in entities
                ]
            }
        ]
    }
    return converted


if __name__ == '__main__':
    data = []
    annotations = load_json_dataset('train.json')

    for annotation in annotations:
        data.append(convert_to_label_studio_format(annotation))

    save_json_dataset(data, 'pre-annotated.json')
