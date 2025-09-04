from typing import List, Optional
import json
import torch
from tqdm.auto import tqdm
import zipfile
import os
import fire
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils.io import load_json
from utils.generation import generate
from scripts.prompt import get_prompt


def run_submission_inference(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        submission_json_path: str,
        run_id: str = "team1_task_2_Method",
        manual: int = 0,
        batch_size: int = 16,
        output_path: str = "submission.json",
):
    raw_examples = load_json(submission_json_path)  # List[dict]
    # build prompts
    # create messages user/assistant
    messages = []
    for ex in raw_examples:
        messages.append([{
            "role": "user",
            "content": get_prompt(ex["en"])
        }])

    prompts = [tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) for msg in messages]
    generaation_config = GenerationConfig(
        do_sample=True,
        temperature=0.2,  # Use temperature to control randomness
        top_p=0.9,  # Use nucleus sampling
        max_new_tokens=128,
        num_beams=3,  # Use beam search
        repetition_penalty=1.3,  # Penalize repetition
        early_stopping=True,  # Stop early if all beams finish
        eos_token_id=tokenizer.eos_token_id,  # Ensure EOS token is set
        pad_token_id=tokenizer.eos_token_id,  # Ensure PAD token is set
    )
    model.generation_config = generaation_config
    print(f"Using generation config: {model.generation_config}")
    # generate all outputs in one go
    generations = generate(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        generation_config=model.generation_config,
        batch_size=batch_size,
    )

    # combine with metadata
    outputs = []
    for ex, gen in zip(raw_examples, generations):
        outputs.append({
            "run_id": run_id,
            "manual": manual,
            "id_en": ex.get("id_en"),
            "en": ex.get("en"),
            "fr": gen,
        })

    # save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(outputs)} records to {output_path}")
    # compress output
    zip_path = output_path.replace(".json", "") + ".zip"
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_path, arcname="prediction.json")
    print(f"Compressed output to {zip_path}")


def main(
        model_path: str,
        test_data: str = "data/task2/joker_pun_translation_2025_test.json",
        output_dir: str = "output/task2",
        batch_size: int = 32,
        manual: int = 0,
        max_seq_length: int = 512,
        load_in_4bit: bool = False,
        use_unsloth: bool = False,
):
    print(f"🚀 Processing model: {model_path} (use_unsloth={use_unsloth})")
    # load model and tokenizer based on user choice
    if use_unsloth:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=not load_in_4bit if not load_in_4bit else False,
        )
        FastLanguageModel.for_inference(model)
    else:
        print("Loading model with Hugging Face transformers")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model_path)
    output_path = os.path.join(output_dir, f"{model_name}_0.2.json")
    # run inference
    run_submission_inference(
        model=model,
        tokenizer=tokenizer,
        submission_json_path=test_data,
        run_id="Skommarkhos_task2_" + model_name,
        manual=manual,
        batch_size=batch_size,
        output_path=output_path,
    )


if __name__ == '__main__':
    fire.Fire(main)
