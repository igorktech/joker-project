from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import json
import os
import random
import fire
import pandas as pd
from prompt import get_prompt


def prepare_cpo_dataset(
        input_file,
        output_file="data/arpo/train.json",
        add_factor=0.5,
        hf_dataset_name="haoranxu/X-ALMA-Preference",
        system_prompt=None,
        add_external_dataset=False,
        seed=42
):
    """
    Process preference pairs JSON data and prepare for ARPO training.

    Args:
        input_file: Path to the JSON with preference pairs (chosen/rejected)
        output_file: Path to save formatted output data
        add_factor: Factor by which to augment training data through duplication
        hf_dataset_name: HuggingFace dataset to use if add_external_dataset=True
        system_prompt: Optional system prompt to include in formatted messages
        add_external_dataset: Whether to include HF preference dataset
        seed: Random seed for reproducibility
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} preference examples from {input_file}")

    ds = Dataset.from_list(data_list)

    # Augment data by duplicating random samples
    if add_factor > 0:
        num_to_add = int(len(ds) * add_factor)
        indices_to_duplicate = np.random.choice(len(ds), size=num_to_add, replace=True)
        samples_to_duplicate = ds.select(indices_to_duplicate)
        ds = concatenate_datasets([ds, samples_to_duplicate])
        print(f"Augmented data to {len(ds)} examples")

    # Add external HF dataset if requested
    if add_external_dataset:
        print(f"Loading external HF preference dataset {hf_dataset_name}...")
        hf = load_dataset(hf_dataset_name)
        # Hardcoded direction filter as "en-fr"
        hf_direction_filter = "en-fr"
        hf_pref = hf['train'].filter(
            lambda x: x['directions'] == hf_direction_filter,
            num_proc=4
        )
        print(f"Loaded {len(hf_pref)} HF preference examples")

        # Format HF dataset to match our structure
        def format_hf_example(example):
            return {
                'prompt': get_prompt(example['source']),
                'chosen': example['chosen'],
                'rejected': example['reject']
            }

        hf_examples = hf_pref.map(format_hf_example)
        # Sample half of examples to avoid overwhelming the dataset
        hf_examples = hf_examples.shuffle(seed=seed).select(range(len(hf_examples) // 2))

        # Combine with our data
        ds = concatenate_datasets([ds, hf_examples])
        print(f"Combined dataset now has {len(ds)} examples")

    # Apply formatting
    def prompt_format(example):
        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": example['prompt']})
        example['prompt'] = prompt_messages
        example['chosen'] = [{"role": "assistant", "content": example['chosen']}]
        example['rejected'] = [{"role": "assistant", "content": example['rejected']}]
        return example

    # Apply formatting
    formatted_data = ds.map(prompt_format)

    # Save dataset to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(formatted_data)
    df.to_json(output_file, orient='records', lines=False, force_ascii=False, indent=4)
    print(f"Saved {len(formatted_data)} examples to {output_file}")

    return formatted_data


if __name__ == "__main__":
    fire.Fire(prepare_cpo_dataset)