import warnings
from typing import Union, List, Any, Dict

import torch
from transformers import DataCollatorForLanguageModeling

import numpy as np


def filter_indices(a, b):
    """
    Фильтрует массив b, оставляя только те элементы, которые следуют сразу после элементов массива a.
    """
    filtered_b = []
    a_len = len(a)
    b_len = len(b)

    # Указатель для массива b
    j = 0

    for i in range(a_len):
        # Ищем индекс элемента из a в b
        while j < b_len and b[j] <= a[i]:
            j += 1
        # Если следующий элемент в b существует, добавляем его в отфильтрованный массив
        if j < b_len:
            filtered_b.append(b[j])
            j += 1  # Переходим к следующему элементу в b

    return filtered_b



class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.response_prompt_template = response_template

        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_prompt_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = self.response_prompt_template

        self.eos_token_id = self.tokenizer.eos_token_id

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_indexes = []
            eos_token_indexes = []

            for idx in torch.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                if (
                    self.response_token_ids
                    == batch["labels"][i][
                        idx : idx + len(self.response_token_ids)
                    ].tolist()
                ):
                    response_token_ids_start_indexes.append(idx.item())

            for idx in torch.where(batch["labels"][i] == self.eos_token_id)[0]:
                eos_token_indexes.append(idx.item())

            eos_token_indexes = filter_indices(
                response_token_ids_start_indexes, eos_token_indexes
            )

            if not response_token_ids_start_indexes or not eos_token_indexes:
                warnings.warn(
                    f"Could not find response key `{self.response_prompt_template}` in the "
                    f"following instance: {self.tokenizer.decode(batch['input_ids'][i])} "
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                new_labels = torch.full_like(batch["labels"][i], self.ignore_index).to(
                    device=batch["labels"][i].device
                )

                for start, end in zip(
                    response_token_ids_start_indexes, eos_token_indexes
                ):
                    new_labels[start : end + 1] = batch["labels"][i, start : end + 1]

                batch["labels"][i] = new_labels

        return batch

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("OpenLLM-France/Lucie-7B-Instruct-v1.1", use_fast=True)
    # Ensure a pad token is set (Llama often uses eos as pad by default)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
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
        {"role": "assistant", "content": "The capital of France is Paris."}],
        [{"role": "user", "content": "What is the capital of Germany?"},
         {"role": "assistant", "content": "The capital of Germany is Berlin."}]]


    examples = [
        tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
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
