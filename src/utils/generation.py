from typing import List, Optional
import json
import torch
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
)

# Helper
def generate(
    prompts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: GenerationConfig,
    batch_size: int = 8,
) -> List[str]:
    """
    Generates only the model’s new text for each prompt, in batches.
    """
    # ensure pad/eos IDs are set
    print("Setting pad/eos token IDs")
    if generation_config.pad_token_id is None:
        print("Set pad token to eos")
        generation_config.pad_token_id = tokenizer.eos_token_id
    if generation_config.eos_token_id is None:
        print("Set eos token to eos")
        generation_config.eos_token_id = [tokenizer.eos_token_id]

    completions: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating completions"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            generations = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # strip off the prompt tokens from each generation
        for input_ids, gen_ids in zip(inputs.input_ids, generations):
            new_tokens = gen_ids[input_ids.shape[-1] :].tolist()
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text.lstrip())

    return completions