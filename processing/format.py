import os

from misc import clean_spacing, DATA_DIR, ROOT_DIR

if __name__ == "__main__":
    print("Cleaning data files...")
    with open(os.path.join(DATA_DIR, 'data.csv'), 'w') as f:
        f.write(clean_spacing('data.csv'))

    with open(os.path.join(DATA_DIR, 'pre-annotated.json'), 'w') as f:
        f.write(clean_spacing('pre-annotated.json'))

    with open(os.path.join(DATA_DIR, 'prompt.txt'), 'w') as f:
        f.write(clean_spacing('prompt.txt'))

    with open(os.path.join(DATA_DIR, 'train.json'), 'w') as f:
        f.write(clean_spacing('train.json'))

    print("Data files cleaned")
