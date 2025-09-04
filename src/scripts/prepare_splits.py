import random
import json
import fire
import os

def prepare_splits(input_file, output_dir, train_size=0.8, val_size=0.1, seed=None):
    """
    Load a JSON file, split it into train/validation/test sets, and save to disk.

    Args:
        input_file (str): Path to the input JSON file.
        output_dir (str): Directory to save the split datasets.
        train_size (float): Proportion of data for training set.
        val_size (float): Proportion of data for validation set.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Shuffle the data to ensure randomness
    if seed is not None:
        random.seed(seed)
        random.shuffle(data)
    else:
        print(f"No seed provided, using default data order.")

    total_size = len(data)
    print(f"Total data size: {total_size}")
    train_end = int(total_size * train_size)
    val_end = int(total_size * (train_size + val_size))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(f"{output_dir}/validation.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    # Only create test file if there's test data
    if test_data:
        with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
        print(f"Data split into train ({len(train_data)}), validation ({len(val_data)}), and test ({len(test_data)}) sets.")
    else:
        print(f"Data split into train ({len(train_data)}) and validation ({len(val_data)}) sets only.")

if __name__ == "__main__":
    fire.Fire(prepare_splits)