import random
import pandas as pd
import os

def create_dataset(save_path):
    categories = {
        'Job': ['doctor', 'teacher', 'engineer', 'nurse', 'carpenter', 'pilot', 'chef', 'scientist', 'lawyer', 'plumber'],
        'Fruit': ['apple', 'banana', 'orange', 'grape', 'mango', 'peach', 'cherry', 'melon', 'pear', 'kiwi'],
        'Vehicle': ['car', 'bike', 'truck', 'scooter', 'bus', 'train', 'airplane', 'boat', 'submarine', 'helicopter'],
        'Animal': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'zebra', 'monkey', 'giraffe', 'bear', 'wolf']
    }

    data = []
    for _ in range(5000):
        category = random.choice(list(categories.keys()))
        word = random.choice(categories[category])
        data.append((word, category))

    df = pd.DataFrame(data, columns=['word', 'type'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset created at {save_path}")

if __name__ == "__main__":
    from config import DATASET_PATH
    create_dataset(DATASET_PATH)