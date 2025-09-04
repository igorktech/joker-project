import json
import os
from tqdm import tqdm
import fire
from openai import OpenAI, base_url
from dotenv import load_dotenv
from src.utils.prompt import get_prompt

def generate_rejected(
        input_file,
        output_file,
        model_name=None,
        base_url=None,
        temperature=0.9,
        top_p=0.6,
        max_tokens=512
):
    """
    Generate rejected translations for DPO training from a dataset.

    Args:
        input_file: Path to JSON dataset with English texts
        output_file: Path to save dataset with rejections
        model_name: Model to use for generating translations (defaults to OPENAI_MODEL env var)
        base_url: Base URL for OpenAI API (defaults to OPENAI_BASE_URL env var)
        temperature: Sampling temperature
        top_p: Top-p sampling value
        max_tokens: Maximum tokens in completion
    """
    # Load environment variables
    load_dotenv()

    # Get configuration from environment with fallbacks
    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "token")
    default_model = os.environ.get("OPENAI_MODEL", "X-ALMA-13B-Group4.Q8_0.gguf")

    # Use provided model_name or fall back to env variable
    model_name = model_name or default_model

    print(f"Using API at: {base_url}")
    print(f"Using model: {model_name}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print(f"Loading dataset from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    results = []
    for item in tqdm(dataset, desc="Generating rejected responses"):
        row = item.copy()

        # Extract English text
        english_text = row.get("en")
        if not english_text:
            print(f"Warning: Missing 'en' field in item: {row}")
            continue

        # Add prompt and chosen response
        row["prompt"] = get_prompt(english_text)
        row["chosen"] = row.get("fr", "")

        # Generate rejected translation
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user",
                     "content": f"Translate this from English to French:\nEnglish: {english_text}\nFrench:"}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            row["rejected"] = response.choices[0].message.content
            row["model"] = model_name
        except Exception as e:
            print(f"Error generating translation for '{english_text}': {e}")
            row["rejected"] = ""

        results.append(row)

    # Save results
    print(f"Saving {len(results)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == "__main__":
    fire.Fire(generate_rejected)