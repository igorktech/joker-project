from datasets import load_dataset, Dataset, concatenate_datasets
import json
import os
import pandas as pd
import fire
from prompt import get_prompt
from transformers import AutoTokenizer


def compose_sft_dataset(
        input_file="data/task2/joker_pun_translation_2025_train.json",
        output_file="data/sft/train.json",
        add_external_dataset=True,
        system_prompt=None,
):
    """
    Process JSON data and optionally join with HF dataset, format, and save as JSON.

    Args:
        input_file: Path to the Joker JSON data file
        output_file: Path to save formatted output data
        add_external_dataset: Whether to include HF parallel corpus (typically True for train, False for validation)
        system_prompt: Optional system prompt to include
    """
    # Load Joker JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} examples from {input_file}")

    ds_joker = Dataset.from_list(data_list)

    # Process data
    if add_external_dataset:
        print("Adding external HF dataset...")
        # Load HF parallel dataset (fr-en subset)
        hf = load_dataset("haoranxu/X-ALMA-Parallel-Data", "fr-en", split="train")

        # Normalize HF examples to joker-like fields 'en' and 'fr'
        def normalize_hf(example):
            return {
                "en": example["translation"]["en"].strip(),
                "fr": example["translation"]["fr"].strip()
            }

        hf_normalized = hf.map(normalize_hf, remove_columns=["translation"])

        # Concatenate
        data = concatenate_datasets([ds_joker, hf_normalized])
        print(f"Combined dataset size: {len(data)}")
    else:
        data = ds_joker
        print(f"Using only Joker dataset: {len(data)} examples")

    # Format messages
    def format_messages(example):
        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.append({"role": "user", "content": get_prompt(example["en"].strip())})
        response = example["fr"].strip()
        prompt_messages.append({"role": "assistant", "content": response})
        example["messages"] = prompt_messages
        return example

    # Apply formatting
    formatted_data = data.map(format_messages)

    # Convert to pandas DataFrame and save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(formatted_data)
    df.to_json(output_file, orient='records', lines=False, force_ascii=False, indent=4)
    print(f"Saved {len(df)} examples to {output_file}")


if __name__ == "__main__":
    fire.Fire(compose_sft_dataset)