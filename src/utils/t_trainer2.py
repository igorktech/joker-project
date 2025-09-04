from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.collators import DataCollatorForCompletionsOnlyLM

tokenizer = AutoTokenizer.from_pretrained("croissantllm/CroissantLLMChat-v0.1",
    # "oktrained/llama3.1_180M_untrained",
    use_fast=True)
dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "completion": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
})
# def frm(ex):
#     ex["text"] =tokenizer.apply_chat_template(
#         ex["prompt"],
#         tokenize=False,
#         # add_generation_prompt=False,
#     )
#     return ex
# dataset = dataset.map(frm, remove_columns=["prompt"], num_proc=1)
# print("Dataset after mapping:\n", dataset[0]['text'])
config = SFTConfig(
    max_seq_length=1024,
    output_dir="./output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    completion_only_loss=True,
)
trainer = SFTTrainer(
    model=AutoModelForCausalLM.from_pretrained("oktrained/llama3.1_180M_untrained"),
    train_dataset=dataset,
    eval_dataset=dataset,
    args=config,
    processing_class=tokenizer,
    # data_collator=DataCollatorForCompletionsOnlyLM(
    #     response_template="<|im_start|>assistant",
    #     tokenizer=tokenizer,
    #     mlm=False,
    #     ignore_index=-100,
    # ),
)

# get first batch and decode it
batch = next(iter(trainer.get_train_dataloader()))
print("Batch", batch)
# Decode the input_ids and labels in the batch
# for i in range(len(batch["input_ids"])):
#     input_text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
#     label_text = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)
#     print(f"Input: {input_text}\nLabel: {label_text}\n")

for i, labels in enumerate(batch["labels"]):
    print(f"\nLabels for example {i}:\n", labels)
    keep = labels != -100
    resp_tokens = batch["input_ids"][i][keep]
    print(f"\nDecoded response for example {i}:\n", tokenizer.decode(resp_tokens))