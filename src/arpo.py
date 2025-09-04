import os
import math
import fire
from transformers import GenerationConfig
from utils.cpo_trainer import CPOTrainer
from utils.cpo_config import CPOConfig
from accelerate import Accelerator
import wandb
import torch
from datasets import Dataset
from utils.seed import set_seed
from utils.io import load_json

os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._dynamo.config.cache_size_limit = 128


def train(config_file: str,
          output_dir: str = None,
          use_unsloth: bool = False  # TODO: Move to config in the future
          ):
    config = load_json(config_file)
    train_file = config.get("train_file")
    eval_file = config.get("eval_file")
    train_list = load_json(train_file)
    eval_list = load_json(eval_file)

    # Use output_dir from config if not provided as argument
    output_dir = output_dir or config.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir must be provided either in config or as argument")

    set_seed(config["seed"])

    max_tokens_count = config["max_tokens_count"]
    max_length = config.get("max_length", max_tokens_count)
    model_name = config["model_name"]

    # Check bf16 support and set appropriate dtype
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if use_unsloth:
        from unsloth import FastLanguageModel
        print("Using Unsloth for ARPO training...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=dtype,
            load_in_8bit=config["load_in_8bit"],
            load_in_4bit=config["load_in_4bit"],
            attn_implementation="sdpa",
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            load_in_8bit=config["load_in_8bit"],
            load_in_4bit=config["load_in_4bit"],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = config["pad_token"]
    tokenizer.eos_token = config["eos_token"]
    tokenizer.bos_token = config["bos_token"]
    tokenizer.padding_side = "right"  # For training
    tokenizer.save_pretrained(output_dir)

    generation_config = config.get("generation_config", None)
    if generation_config:
        print("Using custom generation config:", generation_config)
        generation_config = GenerationConfig(**generation_config)
        print(f"Setting special token: {tokenizer.eos_token_id} for generation config")
        generation_config.eos_token_id = [tokenizer.eos_token_id]
        generation_config.pad_token_id = tokenizer.eos_token_id
    else:
        generation_config = model.generation_config
    generation_config.save_pretrained(output_dir)

    lora_config = config["lora"]
    if lora_config:
        if use_unsloth:
            model = FastLanguageModel.get_peft_model(
                model, **config["lora"], max_seq_length=max_length,
                use_gradient_checkpointing="unsloth",
                random_state=config["seed"],
            )
        else:
            from peft import LoraConfig, TaskType, get_peft_model
            peft_conf = LoraConfig(**config["lora"], task_type=TaskType.CAUSAL_LM)
            model = get_peft_model(model, peft_conf)

    # Prepare datasets
    train_dataset = Dataset.from_list(train_list)
    eval_dataset = Dataset.from_list(eval_list)

    use_nmt_callback = config.get("use_nmt_callback", False)

    trainer_config = config.get("trainer")
    accelerator = Accelerator(log_with=None)  # disable Accelerate’s auto init
    if trainer_config.get("report_to", "wandb") == "wandb":
        if accelerator.is_main_process:
            wandb.init(
                project="joker-pun-translation",
                name=os.path.splitext(os.path.basename(config_file))[0],
                config=config
            )

    training_args = CPOConfig(
        output_dir=output_dir,
        **config["trainer"],
        fp16=not use_bf16,  # Use fp16 if bf16 is not supported
        bf16=use_bf16,  # Use bf16 if supported
        max_length=max_length,
        seed=config["seed"]
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Unsloth flushes special tokens, so we need to set them again
    trainer.processing_class.pad_token = config["pad_token"]
    trainer.processing_class.eos_token = config["eos_token"]
    trainer.processing_class.bos_token = config["bos_token"]

    # Add callback
    if use_nmt_callback:
        try:
            from utils.callbacks import NMTCallback
            from utils.metrics import comet22_metric

            # Calculate total training steps to determine appropriate eval_steps
            num_train_epochs = trainer_config.get("num_train_epochs", 1)
            trainer_eval_steps = trainer_config.get("eval_steps", 2)
            per_device_train_batch_size = trainer_config.get("per_device_train_batch_size", 1)
            gradient_accumulation_steps = trainer_config.get("gradient_accumulation_steps", 1)

            steps_per_epoch = math.ceil(len(train_dataset) /
                                 (per_device_train_batch_size * gradient_accumulation_steps))
            total_training_steps = steps_per_epoch * num_train_epochs

            raw_nmt_step = math.ceil(total_training_steps / 2)

            # Make it a multiple of the Trainer cadence
            nmt_eval_steps = max(1, math.ceil(raw_nmt_step / trainer_eval_steps) * trainer_eval_steps)

            print(f"Training info: {len(train_dataset)} samples")
            print(f"Training steps: {steps_per_epoch} steps/epoch, {total_training_steps} total")
            print(f"NMT evaluation will run every {nmt_eval_steps} training steps (~2 times during training)")

            trainer.add_callback(
                NMTCallback(
                    trainer,
                    generation_config=generation_config,
                    eval_steps=nmt_eval_steps,
                    comet_metric=comet22_metric
                )
            )
        except ImportError as e:
            print(f"Warning: Could not import NMT callback dependencies: {e}")
            print("Continuing without NMT callback...")

    trainer.train()

    trainer.push_to_hub(commit_message="End of training")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
