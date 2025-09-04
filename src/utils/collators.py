import warnings

import torch
from transformers import DataCollatorForLanguageModeling
from typing import Any, Optional, Union


class DataCollatorForCompletionsOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the all completions made by the assistant.

    Args:
        response_template (`Union[str, list[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
            self,
            response_template: Union[str, list[int]],
            *args,
            mlm: bool = False,
            ignore_index: int = -100,
            **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_indexes = []
            eos_token_indexes = []

            for idx in torch.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                        self.response_token_ids
                        == batch["labels"][i][idx: idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_indexes.append(idx+len(self.response_token_ids))

            for idx in torch.where(batch["labels"][i] == self.tokenizer.eos_token_id)[0]:
                eos_token_indexes.append(idx)
            # Filter out EOS token indexes keeping only those that are directly after the corresponding response token
            filtered_eos_token_indexes = []
            for resp_idx in response_token_ids_start_indexes:
                # Find the first EOS token that comes after this response start
                next_eos = next((eos_idx for eos_idx in eos_token_indexes if eos_idx > resp_idx), None)
                if next_eos is not None:
                    filtered_eos_token_indexes.append(next_eos)
            eos_token_indexes = filtered_eos_token_indexes

            if not response_token_ids_start_indexes or not eos_token_indexes:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the following instance: "
                    f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                    "calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                    UserWarning,
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                # Keep tokens for responses, rest of the tokens are ignored
                new_labels = torch.full_like(batch["labels"][i], self.ignore_index).to(batch["labels"][i].device)
                for start_idx, end_idx in zip(response_token_ids_start_indexes, eos_token_indexes):
                    new_labels[start_idx:end_idx+1] = batch["labels"][i][start_idx:end_idx+1]

                batch["labels"][i] = new_labels

        return batch

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        # "croissantllm/CroissantLLMChat-v0.1",
        "OpenLLM-France/Lucie-7B-Instruct-v1.1",
        use_fast=True)
    # Ensure a pad token is set (Llama often uses eos as pad by default)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"#"<|start_header_id|>assistant<|end_header_id|>\n\n"  #"<|im_start|>assistant\n"

    collator = DataCollatorForCompletionsOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        # mlm=False,
        # ignore_index=-100,
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
        tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=False).removeprefix(tokenizer.bos_token)
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