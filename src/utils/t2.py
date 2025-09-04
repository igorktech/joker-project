from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/Lucie-7B-Instruct-v1.1", use_fast=True)
    # Ensure a pad token is set (Llama often uses eos as pad by default)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template="<|start_header_id|>user<|end_header_id|>\n\n",
        tokenizer=tokenizer,
        mlm=False,
        ignore_index=-100,
    )

    # ───────────────────────────────────────────────────────────────────────────────
    # Toy conversation examples
    # ───────────────────────────────────────────────────────────────────────────────
    messages = [[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is the capital of Greece?"},
        {"role": "assistant", "content": "The capital of France is Athens."}
    ],
        [{"role": "user", "content": "What is the capital of Germany?"},
        {"role": "assistant", "content": "The capital of Germany is Berlin."}]]




    examples = [
        tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=False)
        for message in messages
    ]
    print(examples)
    # Convert to tokenized format
    tokenized_examples = tokenizer.batch_encode_plus(
        examples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )

    # The collator expects a list of examples, not the dictionary output of batch_encode_plus
    # Convert the batch dictionary into a list of individual examples
    examples_list = [
        {"input_ids": tokenized_examples["input_ids"][i],
         "attention_mask": tokenized_examples["attention_mask"][i]}
        for i in range(len(tokenized_examples["input_ids"]))
    ]
    print("Tokenized examples:\n", examples_list)



    # ───────────────────────────────────────────────────────────────────────────────
    # Collate & inspect
    # ───────────────────────────────────────────────────────────────────────────────
    batch = collator.torch_call(examples_list)

    print("input_ids:\n", batch["input_ids"])
    print("\nlabels:\n", batch["labels"])


    # Decode the kept response tokens to verify
    for i, labels in enumerate(batch["labels"]):
        keep = labels != collator.ignore_index
        resp_tokens = batch["input_ids"][i][keep]
        print(f"\nDecoded response for example {i}:\n", tokenizer.decode(resp_tokens))
    print("Decoded labels for first example:",)
    print(tokenizer.decode(
        [tokenizer.pad_token_id if x == -100 else x for x in batch["labels"][0]]).replace(
        tokenizer.pad_token, " "))
    # ───────────────────────────────────────────────────────────────────────────────
    # Assertions to auto-check correctness
    # ───────────────────────────────────────────────────────────────────────────────
    for i in range(len(examples)):
        seq = batch["input_ids"][i].tolist()
        start = seq.index(collator.response_token_ids[0])
        lbls = batch["labels"][i]
        # before response must be ignored
        assert torch.all(lbls[:start] == collator.ignore_index)
        # Find all non-ignored positions
        non_ignored = (lbls != collator.ignore_index).nonzero().flatten()
        if len(non_ignored) > 0:
            # Get the last non-ignored position
            eos_pos = non_ignored[-1].item()
            # Check that all non-ignored labels match the input_ids
            assert torch.all(
                lbls[start: eos_pos + 1][lbls[start: eos_pos + 1] != collator.ignore_index] ==
                batch["input_ids"][i][start: eos_pos + 1][lbls[start: eos_pos + 1] != collator.ignore_index]
            )

    print("\n✅ All checks passed")
